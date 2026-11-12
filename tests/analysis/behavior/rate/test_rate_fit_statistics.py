"""The per-bin rate ladder's summary statistics, on fits small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.model_specs import (
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    PER_BIN_HISTORY_COLUMNS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.behavior.per_bin_design import (
    raised_cosine_history_basis,
    reward_kernel_weights,
)
from continuum_swr_replay.analysis.behavior.rate import rate_fit_statistics as stats

NAN = np.nan
ANIMALS = ("a", "b", "c", "d", "e")
WEIGHTS = reward_kernel_weights(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
"""The weight of each reward-basis coefficient at each lag, (n_basis, n_lags). The
first bump has all of its peak at lag 1 and no weight at lag 10; the last has no
weight at lag 1 and its peak at lag 10."""


def _fits(coefs, animals=None, separated=None, converged=None) -> pd.DataFrame:
    """One fit per row of ``coefs`` (the three reward-basis coefficients), each
    its own session; the animals in turn unless given."""
    coefs = np.asarray(coefs, dtype=float)
    n = len(coefs)
    frame = pd.DataFrame(
        {
            "animal": list(animals)
            if animals is not None
            else [ANIMALS[i % 5] for i in range(n)],
            "session_date": [f"d{i}" for i in range(n)],
            "epoch": ["02_r1"] * n,
            "separated_terms": separated if separated is not None else [""] * n,
            "converged": converged if converged is not None else [True] * n,
        }
    )
    for k, term in enumerate(REWARD_HISTORY_COLUMNS):
        frame[f"coef_{term}"] = coefs[:, k]
        # The runaway backstop reads |coef| * sd, so large coefficients on
        # these small-SD features are kept.
        frame[f"sd_{term}"] = 0.1
    return frame


def _first_bump(lag_one) -> np.ndarray:
    """Coefficients on the first bump alone whose kernel at lag 1 is ``lag_one``
    (log rate ratio); that kernel is 0 at lag 10."""
    lag_one = np.asarray(lag_one, dtype=float)
    return np.column_stack(
        [lag_one / WEIGHTS[0, 0], np.zeros_like(lag_one), np.zeros_like(lag_one)]
    )


def _last_bump(lag_ten) -> np.ndarray:
    """Coefficients on the last bump alone whose kernel at lag 10 is ``lag_ten``;
    that kernel is 0 at lag 1 and grows with the lag."""
    lag_ten = np.asarray(lag_ten, dtype=float)
    return np.column_stack(
        [np.zeros_like(lag_ten), np.zeros_like(lag_ten), lag_ten / WEIGHTS[2, -1]]
    )


def _lag_one(coefs) -> np.ndarray:
    return np.asarray(coefs, dtype=float) @ WEIGHTS[:, 0]


# --- the cohort and the baseline rate -------------------------------------------------


def _null_fits(n_stops: dict[str, list[int]], n_bins: dict[str, list[int]]):
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "animal": "a",
                "session_date": session,
                "epoch": "02_r1",
                "n_stops": stops,
                "n_bins": n_bins[outcome][session],
            }
            for outcome, per_session in n_stops.items()
            for session, stops in enumerate(per_session)
        ]
    )


def test_the_frozen_cohort_sums_stops_and_bins_over_sessions():
    populations = ("swr", "non_local", "non_local_swr")
    fits = _null_fits(
        dict.fromkeys(populations, [10, 20]),
        {"swr": [100, 200], "non_local": [90, 180], "non_local_swr": [90, 180]},
    )
    assert stats.frozen_cohort_statistics(fits) == {
        "n_sessions": 2,
        "n_stops": 30,
        "n_bins_non_local": 270,
        "n_bins_swr": 300,
    }


def test_the_frozen_cohort_refuses_populations_on_different_cohorts():
    populations = ("swr", "non_local", "non_local_swr")
    bins = {"swr": [100, 200], "non_local": [90, 180], "non_local_swr": [90, 180]}
    stops = dict.fromkeys(populations, [10, 20]) | {"non_local": [10, 21]}
    with pytest.raises(ValueError, match="differ in cohort"):
        stats.frozen_cohort_statistics(_null_fits(stops, bins))
    with pytest.raises(ValueError, match="differ in cohort"):
        stats.frozen_cohort_statistics(
            _null_fits(
                dict.fromkeys(populations, [10, 20]), bins | {"non_local": [90, 181]}
            )
        )


def _intercepts(rates) -> pd.DataFrame:
    n = len(rates)
    return pd.DataFrame(
        {
            "animal": ["a"] * n,
            "session_date": list(range(n)),
            "epoch": ["02_r1"] * n,
            "coef_Intercept": np.log(rates),
        }
    )


def test_the_baseline_rate_is_exp_of_the_intercept_over_sessions():
    # rates 1, 2, 3, 10: the median is 2.5 and the mean 4
    fits = _intercepts([1.0, 2.0, 3.0, 10.0, NAN])
    result = stats.baseline_rate_statistics(fits)
    assert result["median"] == pytest.approx(2.5)
    assert result["lower_quartile"] == pytest.approx(1.75)  # 1 + 0.75 * (2 - 1)
    assert result["upper_quartile"] == pytest.approx(4.75)  # 3 + 0.25 * (10 - 3)
    assert result["events_per_bin"] == pytest.approx(0.0375)  # 2.5 per s * 15 ms


def test_the_baseline_statistics_refuse_two_fits_of_one_session():
    one = _intercepts([1.0, 2.0, 3.0])
    two = pd.concat([one, one])
    with pytest.raises(ValueError, match="more than one fit for a session"):
        stats.baseline_rate_statistics(two)
    with pytest.raises(pd.errors.MergeError):
        stats.baseline_rate_ratio_statistics(two, one)
    with pytest.raises(pd.errors.MergeError):
        stats.baseline_rate_ratio_statistics(one, two)


def test_the_baseline_ratio_is_within_session():
    # Ratios 2, 3, 4, 10, 1.5 in the five sessions both populations fit: median
    # 3. The ratio of the populations' own medians over those sessions is
    # 8 / 3.
    a = _intercepts([2.0, 9.0, 8.0, 30.0, 6.0, 5.0, 7.0])
    b = _intercepts([1.0, 3.0, 2.0, 3.0, 4.0, NAN])
    result = stats.baseline_rate_ratio_statistics(a, b)
    assert result["n_sessions"] == 5
    assert result["median_ratio"] == pytest.approx(3.0)
    # five differences are too few for a signed-rank p
    assert np.isnan(result["signed_rank_p"])
    assert stats.baseline_rate_ratio_statistics(b, a)["median_ratio"] == pytest.approx(
        1 / 3
    )


# --- the reward kernel ----------------------------------------------------------------


def test_the_reward_gate_leaves_out_a_fit_with_any_separated_reward_term():
    coefs = np.tile([1.0, -0.5, 0.2], (10, 1))
    separated = [""] * 10
    separated[3] = f"{REWARD_HISTORY_COLUMNS[1]}|hist_rcos_0"
    # A separated self-history term alone does not drop the reward kernel.
    separated[4] = "hist_rcos_0"
    fits = _fits(coefs, separated=separated)

    kept = stats.reward_gated_fits(fits)

    assert len(kept) == 9
    assert fits.index[3] not in kept.index
    summary = stats.reward_kernel_statistics(fits)
    assert summary["n_sessions"] == 9 and summary["n_dropped"] == 1


def test_the_kernel_statistics_report_the_kernel_not_the_first_coefficient():
    # A large first coefficient cancelled by the second: exp(c0) is 20x, but
    # the kernel at lag 1 is below 1.
    c0 = np.log(20.0)
    c1 = -c0 * 0.5 / 0.066  # about -22.7: makes 0.32*c0 + 0.066*c1 < 0
    coefs = np.tile([c0, c1, 0.0], (10, 1))
    fits = _fits(coefs, converged=[True] * 9 + [False])

    summary = stats.reward_kernel_statistics(fits)

    assert summary["lag_one_animal_rate_ratio"] == pytest.approx(
        np.exp(_lag_one([c0, c1, 0.0]))
    )
    assert summary["lag_one_animal_rate_ratio"] < 1
    assert summary["n_animals_lag_one_positive"] == 0
    assert summary["n_nonconverged"] == 1


def test_the_lag_one_kernel_over_sessions_and_animals():
    # log lag-1 kernels per session; the seventh fit's last reward term is
    # separated and the eighth has no reward coefficients.
    lag_one = [0.5, 1.0, -0.2, 0.3, 0.4, -0.1, 9.0, NAN]
    fits = _fits(
        _first_bump(lag_one),
        animals=["a", "a", "b", "b", "c", "c", "c", "c"],
        separated=[""] * 6 + [REWARD_HISTORY_COLUMNS[2], ""],
        converged=[True, False, True, True, True, True, False, True],
    )
    result = stats.reward_kernel_statistics(fits)
    assert result["n_sessions"] == 6
    assert result["n_dropped"] == 1  # seven fitted, six kept
    assert result["n_nonconverged"] == 1  # of the kept sessions
    # sorted: -0.2, -0.1, 0.3, 0.4, 0.5, 1.0 -> (e ** 0.3 + e ** 0.4) / 2
    assert result["rate_ratio_lag_1"] == pytest.approx(1.4208417526086367)
    assert result["lag_one_positive_fraction"] == pytest.approx(4 / 6)
    # The negative ranks sum to 1 + 2 = 3; five of the 64 sign patterns have a
    # rank sum of at most 3: two-sided p = 2 * 5 / 64.
    assert result["lag_one_signed_rank_p"] == pytest.approx(0.15625)
    # animal medians 0.75, 0.05, 0.15 -> e ** 0.15
    assert result["lag_one_animal_rate_ratio"] == pytest.approx(1.161834242728283)
    assert result["n_animals"] == 3
    assert result["n_animals_lag_one_positive"] == 3


def test_a_missing_convergence_flag_counts_as_not_converged():
    # one fit converged False, one with no flag; the sixth fit, gated out, has
    # no flag either and is not counted
    fits = _fits(
        _first_bump([0.5, 1.0, -0.2, 0.3, 0.4, 0.6]),
        separated=[""] * 5 + [REWARD_HISTORY_COLUMNS[0]],
        converged=[True, NAN, False, True, True, NAN],
    )
    assert stats.reward_kernel_statistics(fits)["n_nonconverged"] == 2


def test_animals_with_a_negative_median_lag_one_kernel_are_not_counted_positive():
    fits = _fits(
        _first_bump([0.5, 1.0, -0.2, -0.3, 0.4]), animals=["a", "a", "b", "b", "c"]
    )
    result = stats.reward_kernel_statistics(fits)
    assert result["n_animals_lag_one_positive"] == 2
    assert result["lag_one_animal_rate_ratio"] == pytest.approx(
        1.4918246976412703  # medians 0.75, -0.25, 0.4 -> e ** 0.4
    )


def test_the_late_kernel_is_read_from_lag_three_on():
    # First bump only, positive: the kernel falls from lag 1 to nothing at lag
    # 10, so from lag 3 on it is highest at lag 3 and lowest (a ratio of 1) at 10.
    early = stats.reward_kernel_statistics(_fits(_first_bump([0.2, 0.4, 0.6])))
    assert early["rate_ratio_lag_1"] == pytest.approx(1.4918246976412703)  # e ** 0.4
    assert early["rate_ratio_lag_10"] == 1.0
    assert early["late_rate_ratio_min"] == 1.0
    assert early["late_rate_ratio_max"] == early["rate_ratio_lag_3"]
    assert early["rate_ratio_lag_2"] > early["rate_ratio_lag_3"] > 1.0
    # Last bump only, positive: nothing at lag 1, rising to its peak at lag 10.
    late = stats.reward_kernel_statistics(_fits(_last_bump([0.2, 0.4, 0.6])))
    assert late["rate_ratio_lag_1"] == 1.0
    assert late["rate_ratio_lag_10"] == pytest.approx(1.4918246976412703)
    assert late["late_rate_ratio_max"] == late["rate_ratio_lag_10"]
    assert late["late_rate_ratio_min"] == late["rate_ratio_lag_3"]
    assert 1.0 < late["rate_ratio_lag_2"] < late["rate_ratio_lag_3"]
    assert set(early) - {f"rate_ratio_lag_{k}" for k in range(1, 11)} == {
        "n_sessions",
        "n_dropped",
        "n_nonconverged",
        "late_rate_ratio_min",
        "late_rate_ratio_max",
        "lag_one_positive_fraction",
        "lag_one_signed_rank_p",
        "lag_one_animal_rate_ratio",
        "n_animals",
        "n_animals_lag_one_positive",
    }


def test_the_kernel_by_lag_is_the_median_and_quartiles_of_the_gated_sessions():
    # log lag-1 kernels 0.1, 0.2, 0.4, 0.8 in the kept sessions; a fifth
    # session's reward term is separated, a sixth has no reward coefficients
    lag_one = [0.8, 0.1, 9.0, 0.4, NAN, 0.2]
    separated = ["", "", REWARD_HISTORY_COLUMNS[0], "", "", ""]
    kernel = stats.reward_kernel_by_lag(
        _fits(_first_bump(lag_one), separated=separated)
    )
    assert list(kernel.index) == list(range(1, REWARD_HISTORY_N_LAGS + 1))
    assert kernel.index.name == "lag"
    e = np.exp([0.1, 0.2, 0.4, 0.8])
    assert kernel.loc[1, "median"] == pytest.approx((e[1] + e[2]) / 2)  # not e ** 0.3
    assert kernel.loc[1, "lower_quartile"] == pytest.approx(e[0] + 0.75 * (e[1] - e[0]))
    assert kernel.loc[1, "upper_quartile"] == pytest.approx(e[2] + 0.25 * (e[3] - e[2]))
    # the first bump has no weight at lag 10
    assert kernel.loc[REWARD_HISTORY_N_LAGS].tolist() == [1.0, 1.0, 1.0]
    # the medians are the ones the kernel statistics report
    summary = stats.reward_kernel_statistics(
        _fits(_first_bump(lag_one), separated=separated)
    )
    np.testing.assert_array_equal(
        kernel["median"],
        [summary[f"rate_ratio_lag_{lag}"] for lag in kernel.index],
    )


def test_the_kernel_by_lag_reconstructs_each_session_before_the_median():
    """The per-column medians of these coefficients come from three different
    sessions, so a kernel reconstructed from them is no fitted session's."""
    coefs = np.array([[2.0, -0.2, -1.5], [0.1, 1.0, 2.0], [-1.0, -3.0, 0.3]])
    kernel = stats.reward_kernel_by_lag(_fits(coefs))
    per_session = np.exp(coefs @ WEIGHTS)
    np.testing.assert_allclose(kernel["median"], np.median(per_session, axis=0))
    from_median_coefficients = np.exp(np.median(coefs, axis=0) @ WEIGHTS)
    assert abs(kernel.loc[1, "median"] - from_median_coefficients[0]) > 0.05


def test_the_animal_kernels_are_the_median_log_kernel_of_each_animals_kept_sessions():
    # animal a: 0.2, 0.6 -> e ** 0.4 (the median of the two rate ratios would be
    # (e ** 0.2 + e ** 0.6) / 2); animal b: -0.3, and a separated 5.0 left out
    fits = _fits(
        _first_bump([0.2, -0.3, 0.6, 5.0]),
        animals=["a", "b", "a", "b"],
        separated=["", "", "", REWARD_HISTORY_COLUMNS[2]],
    )
    animals = stats.reward_kernel_animal_medians(fits)
    assert list(animals.index) == ["a", "b"]
    assert animals.index.name == "animal"
    assert list(animals.columns) == list(range(1, REWARD_HISTORY_N_LAGS + 1))
    assert animals.loc["a", 1] == pytest.approx(np.exp(0.4))
    assert animals.loc["b", 1] == pytest.approx(np.exp(-0.3))
    assert (animals[REWARD_HISTORY_N_LAGS] == 1.0).all()
    # their median at lag 1 is the animal-balanced value the statistics report
    assert np.exp(np.median(np.log(animals[1]))) == pytest.approx(
        stats.reward_kernel_statistics(fits)["lag_one_animal_rate_ratio"]
    )


def _history_fits(first_bump, separated=None) -> pd.DataFrame:
    """One fit per value: that coefficient on the first self-history bump, 0 on
    the others; NaN for a fit without the block."""
    n = len(first_bump)
    frame = pd.DataFrame(
        {
            "animal": [ANIMALS[i % 5] for i in range(n)],
            "session_date": [f"d{i}" for i in range(n)],
            "epoch": ["02_r1"] * n,
            "separated": separated if separated is not None else [False] * n,
        }
    )
    for k, column in enumerate(PER_BIN_HISTORY_COLUMNS):
        frame[f"coef_{column}"] = (
            first_bump if k == 0 else np.where(np.isnan(first_bump), NAN, 0.0)
        )
    return frame


def test_the_self_history_kernel_by_lag_is_the_median_of_each_sessions_kernel():
    # first-bump coefficients 2.0, -1.0, 0.5, 1.0, a fit without the block, and
    # a separated fit (3.0), which is kept: the self-history kernel is not gated
    fits = _history_fits(
        np.array([2.0, -1.0, NAN, 0.5, 1.0, 3.0]),
        separated=[False, False, False, False, False, True],
    )
    kernel = stats.self_history_kernel_by_lag(fits)
    bump = raised_cosine_history_basis(N_HISTORY_LAGS, N_HISTORY_BASIS)[0]
    assert list(kernel.index) == list(range(1, N_HISTORY_LAGS + 1))
    peak = int(np.argmax(bump))  # an event ends `peak + 1` bins back
    e = np.exp(np.array([-1.0, 0.5, 1.0, 2.0, 3.0]) * bump[peak])
    row = kernel.loc[peak + 1]
    assert row["median"] == pytest.approx(e[2])
    assert row["lower_quartile"] == pytest.approx(e[1])
    assert row["upper_quartile"] == pytest.approx(e[3])
    # an even number of kept sessions: the median averages the middle two
    even = stats.self_history_kernel_by_lag(fits.iloc[:5])
    assert even.loc[peak + 1, "median"] == pytest.approx(
        (np.exp(0.5 * bump[peak]) + np.exp(1.0 * bump[peak])) / 2
    )
    with pytest.raises(ValueError, match="more than one fit for a session"):
        stats.self_history_kernel_by_lag(pd.concat([fits, fits]))


def test_the_paired_test_compares_lag_one_kernels_within_session():
    rng = np.random.default_rng(0)
    n = 20
    # SWR-coupled: larger lag-1 kernel; non-SWR: a larger first coefficient
    # offset by the second, so the coefficient comparison would run the other way.
    swr = np.column_stack([np.full(n, 1.0), np.zeros(n), np.zeros(n)])
    swr[:, 0] += rng.normal(0, 0.05, n)
    no_swr = np.column_stack([np.full(n, 2.0), np.full(n, -12.0), np.zeros(n)])
    no_swr[:, 0] += rng.normal(0, 0.05, n)
    assert (_lag_one(swr) > _lag_one(no_swr)).all()
    assert (swr[:, 0] < no_swr[:, 0]).all()

    paired = stats.paired_lag_one_kernel_statistics(_fits(swr), _fits(no_swr))

    assert paired["n_sessions"] == n
    assert paired["positive_fraction"] == 1.0
    assert paired["n_animals_positive"] == 5


def test_the_paired_test_drops_a_session_gated_in_either_stream():
    coefs = np.tile([1.0, 0.0, 0.0], (10, 1))
    separated = [""] * 10
    separated[2] = REWARD_HISTORY_COLUMNS[0]

    paired = stats.paired_lag_one_kernel_statistics(
        _fits(coefs + 0.1), _fits(coefs, separated=separated)
    )

    assert paired["n_sessions"] == 9


def test_the_paired_difference_is_a_minus_b_over_sessions_and_animals():
    animals = ["a", "a", "b", "b", "c"]
    a = _fits(_first_bump([0.5, 0.2, 0.9, 0.6, -0.2]), animals=animals)
    b = _fits(_first_bump([0.1, 0.4, 0.3, 0.1, 0.1]), animals=animals)
    # differences 0.4, -0.2, 0.6, 0.5, -0.3; animal medians 0.1, 0.55, -0.3
    result = stats.paired_lag_one_kernel_statistics(a, b)
    assert result["n_sessions"] == 5
    assert result["positive_fraction"] == 0.6
    assert result["n_animals"] == 3
    assert result["n_animals_positive"] == 2
    # five differences are too few for a signed-rank p
    assert np.isnan(result["signed_rank_p"])
    reverse = stats.paired_lag_one_kernel_statistics(b, a)
    assert reverse["positive_fraction"] == 0.4
    assert reverse["n_animals_positive"] == 1


def test_the_reward_kernel_refuses_two_fits_of_one_session():
    one = _fits(_first_bump([0.5, 0.2, 0.9, 0.6, -0.2]))
    two = pd.concat([one, one])
    with pytest.raises(ValueError, match="more than one fit for a session"):
        stats.reward_kernel_statistics(two)
    with pytest.raises(ValueError, match="more than one fit for a session"):
        stats.paired_lag_one_kernel_statistics(one, two)


# --- each rung's fits -----------------------------------------------------------------


def test_ladder_fit_statistics_have_one_row_per_rung_and_population():
    fits = pd.DataFrame(
        {
            "model": ["M5"] * 4 + ["M0"] * 2,
            "outcome": ["swr"] * 4 + ["non_local"] * 2,
            "coef_Intercept": [1.0, 2.0, NAN, 3.0, 0.5, 0.5],
            "aic": [10.0, 20.0, NAN, np.inf, 7.0, 9.0],
            "separated": [True, False, False, None, False, False],
            "cv_bits_per_event": [0.5, NAN, 1.5, -np.inf, 0.25, 0.75],
            "ks_within_band": [True, True, False, None, False, False],
            "ks_statistic": [0.1, 0.4, 0.2, NAN, 0.3, 0.5],
            "dispersion": [0.5, 0.75, 0.25, 4.0, NAN, 1.0],
        }
    )
    table = stats.ladder_fit_statistics(fits)
    assert list(table.index.names) == ["model", "outcome"]
    assert set(table.index) == {("M5", "swr"), ("M0", "non_local")}
    assert table.loc[("M5", "swr")].to_dict() == {
        "n_fitted": 3,
        "n_finite_aic": 2,
        "aic_median": 20.0,  # of 10, 20 and an infinite one
        "separated_fraction": 0.25,  # one of all four fits
        "n_cv_finite": 2,
        "cv_bits_median": 1.0,
        "cv_not_finite_fraction": 0.5,  # two of all four fits
        # two of the three fits that carry the flag, not of all four
        "ks_band_fraction": pytest.approx(2 / 3),
        "ks_statistic_median": 0.2,  # of the three fits that carry one
        "dispersion_median": 0.625,  # of all four (their mean is 1.375)
    }
    assert table.loc[("M0", "non_local")].to_dict() == {
        "n_fitted": 2,
        "n_finite_aic": 2,
        "aic_median": 8.0,
        "separated_fraction": 0.0,
        "n_cv_finite": 2,
        "cv_bits_median": 0.5,
        "cv_not_finite_fraction": 0.0,
        "ks_band_fraction": 0.0,
        "ks_statistic_median": 0.4,
        "dispersion_median": 1.0,  # the one fit that carries one
    }


def test_the_arrival_window_shares_are_of_the_frozen_cohort():
    cohort = {
        "n_bins_frozen": 800,
        "n_event_bins_frozen": 50,
        "n_bins_dropped_arrival_clock": 200,
        "n_bins_arrival_window": 600,
        "n_event_bins_arrival_window": 40,
    }
    assert stats.arrival_window_statistics(cohort) == {
        "event_share": 0.8,
        "bin_share": 0.75,
        "n_bins_before": 1000,
        "dropped_bin_fraction": 0.2,
    }


# --- the fitted time course -----------------------------------------------------------

STOP_SPLINE = "coef_bs(tc, knots=(1.0, 2.0, 3.0), lower_bound=0.0, upper_bound=4.0)"
ARRIVAL_SPLINE = (
    "coef_bs(time_since_last_well_arrival_s, knots=(1.0, 2.0, 4.0, 6.0), "
    "lower_bound=0.0, upper_bound=8.0)"
)


def _time_fits(spline: str, n_basis: int, intercepts, bumps=None, separated=None):
    """One time-rung fit per intercept. ``bumps`` maps a basis index to its
    coefficient (in every session); the others are 0. The basis columns are
    stored out of order, beside an interaction with the spline."""
    n = len(intercepts)
    frame = pd.DataFrame(
        {
            "animal": ["a"] * n,
            "session_date": list(range(n)),
            "epoch": ["02_r1"] * n,
            "coef_Intercept": np.asarray(intercepts, dtype=float),
            "separated": separated if separated is not None else [False] * n,
        }
    )
    for k in [*range(n_basis - 1, -1, -2), *range(n_basis - 2, -1, -2)]:
        frame[f"{spline}[{k}]"] = (bumps or {}).get(k, 0.0)
    frame[f"coef_reward_history_rcos_0:{spline.removeprefix('coef_')}[0]"] = 5.0
    frame[f"{spline}[0]:reward_history_rcos_0[1]"] = 5.0
    return frame


def test_a_flat_time_course_is_the_median_session_rate():
    fits = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0, 4.0]))
    grid, curve, n_curves = stats.fitted_time_course(fits, "stop")
    assert grid.shape == curve.shape == (401,)
    assert grid[0] == 1.0 and grid[-1] == 3.0 and grid[200] == 2.0
    np.testing.assert_allclose(curve, 2.0)
    assert n_curves == 3


def test_the_time_course_is_the_fitted_spline_of_each_clock():
    # A cubic spline reproduces a straight line when each basis column's
    # coefficient is the mean of its three inner knots. Stop clock, knots
    # 0,0,0,0,1,2,3,4,4,4,4: 1/3, 1, 2, 3, 11/3, 4 (the first column, at 0, is
    # the dropped intercept). Arrival clock, knots 0,0,0,0,1,2,4,6,8,8,8,8:
    # 1/3, 1, 7/3, 4, 6, 22/3, 8. With those coefficients times log 2 the rate
    # is 2 ** t on either clock, whatever the knots.
    log2 = np.log(2.0)
    stop = _time_fits(
        STOP_SPLINE,
        6,
        [0.0],
        bumps=dict(enumerate(log2 * np.array([1 / 3, 1, 2, 3, 11 / 3, 4]))),
    )
    grid, curve, _ = stats.fitted_time_course(stop, "stop")
    np.testing.assert_allclose(curve, 2.0**grid)
    arrival = _time_fits(
        ARRIVAL_SPLINE,
        7,
        [0.0],
        bumps=dict(enumerate(log2 * np.array([1 / 3, 1, 7 / 3, 4, 6, 22 / 3, 8]))),
    )
    grid, curve, _ = stats.fitted_time_course(arrival, "arrival")
    assert grid[0] == 1.0 and grid[-1] == 6.0
    np.testing.assert_allclose(curve, 2.0**grid)


def test_the_time_course_leaves_out_separated_and_non_finite_sessions():
    # kept: rates 1, 2, 4. Left out: a separated session, one with no
    # intercept, one whose rate overflows, and one missing a spline coefficient.
    fits = _time_fits(
        STOP_SPLINE,
        6,
        [*np.log([1.0, 2.0, 4.0]), np.log(100.0), NAN, 1e4, np.log(200.0)],
        separated=[False, False, None, True, False, False, False],
    )
    fits.loc[6, f"{STOP_SPLINE}[3]"] = NAN
    _, curve, n_curves = stats.fitted_time_course(fits, "stop")
    np.testing.assert_allclose(curve, 2.0)
    assert n_curves == 3


def test_no_usable_session_is_no_time_course():
    # Every session is separated, has no intercept or overflows: a median over
    # no curve would put the peak at the start of the interval.
    separated = _time_fits(
        STOP_SPLINE, 6, [0.0, NAN, 1e4], separated=[True, False, False]
    )
    with pytest.raises(ValueError, match="no session has a usable stop-clock"):
        stats.fitted_time_course(separated, "stop")
    flat_arrival = _time_fits(ARRIVAL_SPLINE, 7, np.log([1.0, 2.0]))
    with pytest.raises(ValueError, match="no session has a usable stop-clock"):
        stats.clock_shape_statistics(separated, flat_arrival)
    none_arrival = _time_fits(ARRIVAL_SPLINE, 7, [0.0], separated=[True])
    flat_stop = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0]))
    with pytest.raises(ValueError, match="no session has a usable arrival-clock"):
        stats.clock_shape_statistics(flat_stop, none_arrival)


def test_the_time_course_refuses_two_fits_of_one_session():
    fits = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0, 4.0]))
    with pytest.raises(ValueError, match="more than one fit for a session"):
        stats.fitted_time_course(pd.concat([fits, fits]), "stop")


def test_a_peak_inside_the_interval_is_located_and_one_at_its_end_is_not():
    # Stop clock: knots at 1, 2, 3 on [0, 4] are uniform, so the middle cubic
    # B-spline (the third of the six columns: the first of the seven is the
    # dropped intercept) is symmetric about 2 s and peaks there.
    located = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0]), bumps={2: 1.0})
    # The fifth column's B-spline spans 2 to 4 s and is still rising at 3 s.
    rising = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0, 3.0]), bumps={4: 1.0})
    flat_arrival = _time_fits(ARRIVAL_SPLINE, 7, np.log([1.0, 2.0, 4.0, 8.0]))

    result = stats.clock_shape_statistics(located, flat_arrival)
    assert result["time_peak_s"] == 2.0
    assert result["time_peak_edge"] == 0
    assert result["time_peak_sessions"] == 2
    # a flat course has its first point as argmax: the start of the arrival
    # clock's interval, 1 s
    assert result["arrival_peak_s"] == 1.0
    assert result["arrival_peak_edge"] == 1
    assert result["arrival_peak_sessions"] == 4
    # each clock's interval runs between its own outermost interior knots
    assert (result["time_interval_start_s"], result["time_interval_end_s"]) == (1, 3)
    assert (result["arrival_interval_start_s"], result["arrival_interval_end_s"]) == (
        1,
        6,
    )

    result = stats.clock_shape_statistics(rising, flat_arrival)
    assert result["time_peak_s"] == 3.0
    assert result["time_peak_edge"] == 1
    assert result["time_peak_sessions"] == 3


def test_a_peak_on_the_arrival_clock_is_located_on_its_own_knots():
    # A cubic spline reproduces t * (10 - t), which peaks at 5 s, when each
    # column's coefficient is 10 times the mean of its three inner knots less
    # the mean of their pairwise products. Arrival knots 0,0,0,0,1,2,4,6,8,8,8,8:
    # means 1/3, 1, 7/3, 4, 6, 22/3, 8 and pairwise means 0, 2/3, 14/3, 44/3,
    # 104/3, 160/3, 64, so 10/3, 28/3, 56/3, 76/3, 76/3, 20, 16.
    coefficients = np.array([10 / 3, 28 / 3, 56 / 3, 76 / 3, 76 / 3, 20, 16]) / 10
    peaked = _time_fits(
        ARRIVAL_SPLINE, 7, np.log([1.0, 3.0]), bumps=dict(enumerate(coefficients))
    )
    flat_stop = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0, 4.0]))
    result = stats.clock_shape_statistics(flat_stop, peaked)
    assert result["arrival_peak_s"] == pytest.approx(5.0)
    assert result["arrival_peak_edge"] == 0
    assert result["arrival_peak_sessions"] == 2
    assert result["time_peak_s"] == 1.0  # flat: its first point
    assert result["time_peak_edge"] == 1


def test_the_ks_distance_compares_shapes_on_relative_position():
    flat_stop = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0, 4.0]))
    flat_arrival = _time_fits(ARRIVAL_SPLINE, 7, np.log([3.0, 5.0]))
    # Two flat courses have the same shape whatever their rates and intervals.
    assert stats.clock_shape_statistics(flat_stop, flat_arrival)[
        "clock_ks"
    ] == pytest.approx(0.0, abs=1e-12)
    # A stop-clock rate of 2 ** t (coefficients log 2 times 1/3, 1, 2, 3, 11/3,
    # 4) has the cumulative shape (2 ** t - 2) / 6 on [1, 3] s; a flat course
    # has (t - 1) / 2. They are furthest apart where 2 ** t = 3 / log 2, at
    # t = 2.1137 s: 0.556865 - 0.388014 = 0.168851.
    doubling = _time_fits(
        STOP_SPLINE,
        6,
        np.log([1.0, 2.0, 3.0]),
        bumps=dict(enumerate(np.log(2.0) * np.array([1 / 3, 1, 2, 3, 11 / 3, 4]))),
    )
    ks = stats.clock_shape_statistics(doubling, flat_arrival)["clock_ks"]
    assert ks == pytest.approx(0.168851, abs=2e-5)


def test_the_cumulative_shape_is_the_trapezoid_area_scaled_to_one():
    # areas of the two intervals: 1 * (1 + 1) / 2 = 1 and 1 * (1 + 3) / 2 = 2
    cdf = stats._shape_cdf(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 3.0]))
    np.testing.assert_allclose(cdf, [0.0, 1 / 3, 1.0])
    # unequal steps: areas 2 * 1 = 2 and 1 * 2 = 2
    cdf = stats._shape_cdf(np.array([0.0, 2.0, 3.0]), np.array([1.0, 1.0, 3.0]))
    np.testing.assert_allclose(cdf, [0.0, 0.5, 1.0])
