"""The per-bin rate ladder's summary statistics, on fits small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import rate_fit_statistics as stats
from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import reward_kernel_weights

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


def test_the_baseline_rate_is_exp_of_the_intercept_over_sessions():
    fits = pd.DataFrame({"coef_Intercept": [*np.log([1.0, 2.0, 3.0, 4.0]), NAN]})
    result = stats.baseline_rate_statistics(fits)
    assert result["median"] == pytest.approx(2.5)
    assert result["lower_quartile"] == pytest.approx(1.75)
    assert result["upper_quartile"] == pytest.approx(3.25)
    assert result["events_per_bin"] == pytest.approx(0.0375)  # 2.5 per s * 15 ms


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


def test_the_baseline_ratio_is_within_session():
    # ratios 2, 4, 4, 8 in the four sessions both populations fit: median 4
    a = _intercepts([2.0, 4.0, 8.0, 16.0, 5.0, 7.0])
    b = _intercepts([1.0, 1.0, 2.0, 2.0, NAN])
    result = stats.baseline_rate_ratio_statistics(a, b)
    assert result["n_sessions"] == 4
    assert result["median_ratio"] == pytest.approx(4.0)
    # all four differences positive: two-sided exact p = 2 / 2 ** 4
    assert result["signed_rank_p"] == pytest.approx(0.125)
    assert stats.baseline_rate_ratio_statistics(b, a)["median_ratio"] == pytest.approx(
        0.25
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


def test_animals_with_a_negative_median_lag_one_kernel_are_not_counted_positive():
    fits = _fits(
        _first_bump([0.5, 1.0, -0.2, -0.3, 0.4]), animals=["a", "a", "b", "b", "c"]
    )
    result = stats.reward_kernel_statistics(fits)
    assert result["n_animals_lag_one_positive"] == 2
    assert result["lag_one_animal_rate_ratio"] == pytest.approx(
        1.4918246976412703  # medians 0.75, -0.25, 0.4 -> e ** 0.4
    )


# The last-bump kernels are exactly 0 at lag 1, which the signed-rank test cannot rank.
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
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
    # The negative ranks sum to 1 + 2 = 3; five of the 32 sign patterns have a
    # rank sum of at most 3: two-sided p = 2 * 5 / 32.
    assert result["signed_rank_p"] == pytest.approx(0.3125)
    reverse = stats.paired_lag_one_kernel_statistics(b, a)
    assert reverse["positive_fraction"] == 0.4
    assert reverse["n_animals_positive"] == 1


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
            "ks_within_band": [True, True, False, True, False, False],
        }
    )
    table = stats.ladder_fit_statistics(fits)
    assert list(table.index.names) == ["model", "outcome"]
    assert set(table.index) == {("M5", "swr"), ("M0", "non_local")}
    assert table.loc[("M5", "swr")].to_dict() == {
        "n_fitted": 3,
        "n_finite_aic": 2,
        "aic_median": 20.0,  # of 10, 20 and an infinite one
        "separated_fraction": 0.25,
        "n_cv_finite": 2,
        "cv_bits_median": 1.0,
        "cv_not_finite_fraction": 0.5,
        "ks_band_fraction": 0.75,
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
    grid, curve, n_curves = stats.fitted_time_course(fits, "tc", (1.0, 2.0, 3.0), 4.0)
    assert grid.shape == curve.shape == (401,)
    assert grid[0] == 1.0 and grid[-1] == 3.0 and grid[200] == 2.0
    np.testing.assert_allclose(curve, 2.0)
    assert n_curves == 3


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
    _, curve, n_curves = stats.fitted_time_course(fits, "tc", (1.0, 2.0, 3.0), 4.0)
    np.testing.assert_allclose(curve, 2.0)
    assert n_curves == 3


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

    result = stats.clock_shape_statistics(rising, flat_arrival)
    assert result["time_peak_s"] == 3.0
    assert result["time_peak_edge"] == 1
    assert result["time_peak_sessions"] == 3


def test_the_ks_distance_compares_shapes_on_relative_position():
    flat_stop = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0, 4.0]))
    flat_arrival = _time_fits(ARRIVAL_SPLINE, 7, np.log([3.0, 5.0]))
    # Two flat courses have the same shape whatever their rates and intervals.
    assert stats.clock_shape_statistics(flat_stop, flat_arrival)[
        "clock_ks"
    ] == pytest.approx(0.0, abs=1e-12)
    # A course that rises through its interval has its mass late: its
    # cumulative shape lies below the flat one's everywhere inside, so the
    # signed difference is negative and the distance is its size.
    rising = _time_fits(STOP_SPLINE, 6, np.log([1.0, 2.0, 3.0]), bumps={4: 3.0})
    ks = stats.clock_shape_statistics(rising, flat_arrival)["clock_ks"]
    assert 0.05 < ks < 1.0


def test_the_cumulative_shape_is_the_trapezoid_area_scaled_to_one():
    # areas of the two intervals: 1 * (1 + 1) / 2 = 1 and 1 * (1 + 3) / 2 = 2
    cdf = stats._shape_cdf(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 3.0]))
    np.testing.assert_allclose(cdf, [0.0, 1 / 3, 1.0])
    # unequal steps: areas 2 * 1 = 2 and 1 * 2 = 2
    cdf = stats._shape_cdf(np.array([0.0, 2.0, 3.0]), np.array([1.0, 1.0, 3.0]))
    np.testing.assert_allclose(cdf, [0.0, 0.5, 1.0])
