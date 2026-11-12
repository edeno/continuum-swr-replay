"""The behavior summary's fit values hold the statistic their key names, on hand-made fits.

Each population, rung and term is given its own number, so a value read from
another population, rung or term, or the lowest read as the highest, fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.model_specs import (
    PER_BIN_HISTORY_COLUMNS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import reward_kernel_weights
from continuum_swr_replay.reported import behavior

NAN = np.nan
ANIMALS = ("a", "a", "b", "c", "d", "e")
"""Six sessions of five animals: the median over sessions is not the median
over the animals' own medians."""

POPULATIONS = {
    "swr": {
        "rate": 1.0,
        "lag_one": 1.5,
        "lag_ten": 1.1,
        "stop_peak": 1.5,
        "arrival_peak": 2.0,
        "n_separated": 0,
        "window_events": 50,
    },
    "non_local": {
        "rate": 2.0,
        "lag_one": 2.0,
        "lag_ten": 1.2,
        "stop_peak": 2.0,
        "arrival_peak": 3.0,
        "n_separated": 1,
        "window_events": 60,
    },
    "non_local_swr": {
        "rate": 4.0,
        "lag_one": 3.0,
        "lag_ten": 1.3,
        "stop_peak": 2.5,
        "arrival_peak": 4.0,
        "n_separated": 2,
        "window_events": 70,
    },
    "non_local_no_swr": {
        "rate": 8.0,
        "lag_one": 0.5,
        "lag_ten": 1.4,
        "stop_peak": 3.5,
        "arrival_peak": 5.0,
        "n_separated": 3,
        "window_events": 80,
    },
}
"""Per population: the constant rate of its first session (events/s), its reward
kernel at lags 1 and 10 (rate ratios), where its rate peaks on each clock (s),
how many of its arrival-clock time fits are separated, and how many of its 100
event bins the arrival window keeps."""

AIC = {
    "M0_const": 100.0,
    "M1_rew": 97.0,
    "M3r_histcos": 95.0,
    "M5_time": 90.0,
    "M6r_time_histcos": 80.0,
    "M7r_time_histcos_rew": 70.0,
    "M9r_rew_x_time": 65.0,
    "M0a_const_arrival": 60.0,
    "M5a_time_arrival": 55.0,
    "M7ra_time_arrival_histcos_rew": 50.0,
}
"""Each rung's AIC, in every session and population."""

STOP_SPLINE = "coef_bs(tc, knots=(1.0, 2.0, 3.0), lower_bound=0.0, upper_bound=4.0)"
ARRIVAL_SPLINE = (
    "coef_bs(time_since_last_well_arrival_s, knots=(1.0, 2.0, 4.0, 6.0), "
    "lower_bound=0.0, upper_bound=8.0)"
)
# A cubic spline is t * (2 * peak - t) when each basis column's coefficient is
# 2 * peak times the mean of its three inner knots, less the mean of their
# pairwise products. Stop clock, knots 0,0,0,0,1,2,3,4,4,4,4; arrival clock,
# knots 0,0,0,0,1,2,4,6,8,8,8,8 (the first column is the dropped intercept).
STOP_KNOTS = (
    (1 / 3, 0.0),
    (1.0, 2 / 3),
    (2.0, 11 / 3),
    (3.0, 26 / 3),
    (11 / 3, 40 / 3),
    (4.0, 16.0),
)
ARRIVAL_KNOTS = (
    (1 / 3, 0.0),
    (1.0, 2 / 3),
    (7 / 3, 14 / 3),
    (4.0, 44 / 3),
    (6.0, 104 / 3),
    (22 / 3, 160 / 3),
    (8.0, 64.0),
)
WEIGHTS = reward_kernel_weights(REWARD_HISTORY_N_LAGS, REWARD_HISTORY_N_BASIS)
"""The weight of each reward-basis coefficient at each lag: the first has no
weight at lag 10, the last none at lag 1."""


def _arrival_cohort(window_events: dict[str, int]) -> dict[str, dict[str, int]]:
    """The runner's arrival-clock record: 100 event bins and 1000 bins frozen
    per population, 800 bins in the window, 250 bins removed by the clock."""
    return {
        outcome: {
            "n_bins_frozen": 1000,
            "n_event_bins_frozen": 100,
            "n_bins_dropped_arrival_clock": 250,
            "n_stops_dropped_arrival_clock": 7,
            "n_bins_arrival_window": 800,
            "n_event_bins_arrival_window": events,
        }
        for outcome, events in window_events.items()
    }


def _per_bin_fits() -> pd.DataFrame:
    """Every rung and population of :data:`POPULATIONS` in six sessions.

    Session ``s`` (from 1) of a population has ``s`` times its rate, in every
    rung. Its reward kernel is the same in every session. Its log rate is
    ``t * (2 * peak - t)`` on each clock. The swr population has 120 bins a
    session, the others 100; every one has 10 stops.
    """
    rows = []
    for outcome, population in POPULATIONS.items():
        for model, aic in AIC.items():
            for session, animal in enumerate(ANIMALS):
                row = {
                    "model": model,
                    "outcome": outcome,
                    "animal": animal,
                    "session_date": session,
                    "epoch": "02_r1",
                    "coef_Intercept": np.log(population["rate"] * (session + 1)),
                    "aic": aic,
                    "cv_bits_per_event": aic / 100,
                    "ks_within_band": True,
                    "ks_statistic": 0.1,
                    "dispersion": 1.0,
                    "separated": model == "M5a_time_arrival"
                    and session < population["n_separated"],
                    "separated_terms": "",
                    "converged": True,
                    "n_stops": 10,
                    "n_bins": 120 if outcome == "swr" else 100,
                    f"coef_{REWARD_HISTORY_COLUMNS[0]}": np.log(population["lag_one"])
                    / WEIGHTS[0, 0],
                    f"coef_{REWARD_HISTORY_COLUMNS[1]}": 0.0,
                    f"coef_{REWARD_HISTORY_COLUMNS[2]}": np.log(population["lag_ten"])
                    / WEIGHTS[2, -1],
                }
                row |= {f"sd_{column}": 0.1 for column in REWARD_HISTORY_COLUMNS}
                row |= {f"coef_{column}": 0.1 for column in PER_BIN_HISTORY_COLUMNS}
                for spline, knots, peak in (
                    (STOP_SPLINE, STOP_KNOTS, population["stop_peak"]),
                    (ARRIVAL_SPLINE, ARRIVAL_KNOTS, population["arrival_peak"]),
                ):
                    for k, (mean, product) in enumerate(knots):
                        row[f"{spline}[{k}]"] = 2 * peak * mean - product
                rows.append(row)
    fits = pd.DataFrame(rows)
    fits.attrs["arrival_clock_cohort"] = _arrival_cohort(
        {outcome: p["window_events"] for outcome, p in POPULATIONS.items()}
    )
    return fits


def _all_stops_fits() -> pd.DataFrame:
    """The same fits, of which the arrival window keeps 90 event bins of 100."""
    fits = _per_bin_fits()
    fits.attrs["arrival_clock_cohort"] = _arrival_cohort(dict.fromkeys(POPULATIONS, 90))
    return fits


def _values(values: list) -> dict[str, float]:
    return {value.key: value.value for value in values}


# --- the per-bin rate ladder ----------------------------------------------------------


def test_per_bin_values_read_each_population_s_own_statistic():
    values = _values(
        behavior.per_bin_values(_per_bin_fits(), _all_stops_fits(), "source")
    )
    assert values["rate_sessions"] == 6
    assert values["rate_stops"] == 60
    assert values["rate_bins"] == 600
    assert values["rate_bins_swr"] == 720
    # a population's rates are 1 to 6 times its own: median 3.5 times
    assert values["rate_baseline_swr"] == pytest.approx(3.5)
    assert values["rate_baseline_non_local_no_swr"] == pytest.approx(28.0)
    assert values["rate_kernel_lag_one_swr"] == pytest.approx(1.5)
    assert values["rate_kernel_lag_one_non_local_no_swr"] == pytest.approx(0.5)
    assert values["rate_kernel_lag_one_animal_non_local"] == pytest.approx(2.0)
    assert values["rate_kernel_lag_ten_non_local_swr"] == pytest.approx(1.3)
    assert values["rate_kernel_lag_one_animals_positive_swr"] == 5
    assert values["rate_kernel_lag_one_animals_positive_non_local_no_swr"] == 0
    assert values["rate_kernel_lag_one_positive_fraction_non_local_no_swr"] == 0.0
    # the first basis coefficient is log(lag-1 kernel) over its weight at lag 1
    assert values["rate_immediate_session_non_local"] == pytest.approx(
        2.0 ** (1 / WEIGHTS[0, 0])
    )
    assert values["rate_immediate_animals_positive_non_local_no_swr"] == 0
    assert values["rate_immediate_ratio_swr_to_no_swr"] == pytest.approx(
        (3.0 / 0.5) ** (1 / WEIGHTS[0, 0])
    )
    # child minus parent: M1_rew - M0_const, and the full rung - M6r_time_histcos
    assert values["rate_marginal_reward_delta_aic_swr"] == -3.0
    assert values["rate_conditional_reward_delta_aic_swr"] == -10.0
    for word, rung in (
        ("const", "M0_const"),
        ("time", "M5_time"),
        ("full", "M7r_time_histcos_rew"),
        ("reward_by_time", "M9r_rew_x_time"),
    ):
        assert values[f"rate_ladder_aic_{word}"] == AIC[rung]
        assert values[f"rate_ladder_cv_bits_{word}"] == pytest.approx(AIC[rung] / 100)


def test_per_bin_values_read_each_clock_s_time_course():
    values = _values(
        behavior.per_bin_values(_per_bin_fits(), _all_stops_fits(), "source")
    )
    assert values["rate_time_peak_s_non_local"] == pytest.approx(2.0)
    assert values["rate_arrival_peak_s_non_local"] == pytest.approx(3.0)
    assert values["rate_time_peak_edge_non_local"] == 0
    # The last population peaks at 3.5 s on the stop clock, past the end of
    # its interval at 3 s: its peak is bounded there, not located.
    assert values["rate_time_peak_s_non_local_no_swr"] == pytest.approx(3.0)
    assert values["rate_time_peak_edge_non_local_no_swr"] == 1
    assert values["rate_arrival_peak_edge_non_local_no_swr"] == 0
    assert values["rate_time_peak_sessions_non_local_no_swr"] == 6
    # three of its six arrival-clock fits are separated
    assert values["rate_arrival_peak_sessions_non_local_no_swr"] == 3
    # The ranges over populations: located stop-clock peaks 1.5, 2 and 2.5 s,
    # arrival-clock peaks 2 to 5 s, separated shares 0 to 3 of 6.
    assert values["rate_time_peak_edge_count"] == 1
    assert values["rate_time_peak_s_low"] == pytest.approx(1.5)
    assert values["rate_time_peak_s_high"] == pytest.approx(2.5)
    assert values["rate_arrival_peak_edge_count"] == 0
    assert values["rate_arrival_peak_s_low"] == pytest.approx(2.0)
    assert values["rate_arrival_peak_s_high"] == pytest.approx(5.0)
    assert values["rate_arrival_separated_fraction_low"] == 0.0
    assert values["rate_arrival_separated_fraction_high"] == 0.5
    assert values["rate_arrival_separated_fraction_non_local"] == pytest.approx(1 / 6)


def test_per_bin_values_read_the_paired_kernel_and_the_all_stops_fit():
    values = _values(
        behavior.per_bin_values(_per_bin_fits(), _all_stops_fits(), "source")
    )
    # SWR-coupled (3) minus non-SWR (0.5) lag-1 kernel: positive in every session
    assert values["rate_paired_sessions"] == 6
    assert values["rate_paired_positive_fraction"] == 1.0
    assert values["rate_paired_animals_positive"] == 5
    assert values["rate_arrival_event_share_swr"] == 0.5
    assert values["rate_arrival_event_share_non_local_no_swr"] == 0.8
    assert values["rate_allstops_arrival_event_share_swr"] == 0.9
    assert values["rate_allstops_kernel_lag_one_no_swr"] == pytest.approx(0.5)
    assert values["rate_allstops_kernel_lag_one_animals_positive_no_swr"] == 0
    assert values["rate_allstops_immediate_animals_positive_swr"] == 5
    assert values["rate_allstops_immediate_animals_positive_non_local_no_swr"] == 0


# The second reward coefficient is exactly 0, which the signed-rank test cannot rank.
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_per_bin_detail_values_read_each_population_rung_and_term():
    in_transit = pd.DataFrame()
    in_transit.attrs["arrival_clock_cohort"] = _arrival_cohort(
        {"swr": 20, "non_local": 30, "non_local_swr": 40, "non_local_no_swr": 50}
    )
    values = _values(
        behavior.per_bin_detail_values(
            _per_bin_fits(), _all_stops_fits(), in_transit, "source"
        )
    )
    # rates 1 to 6 times the population's: quartiles 2.25 and 4.75 times
    assert values["rate_baseline_lower_quartile_swr"] == pytest.approx(2.25)
    assert values["rate_baseline_upper_quartile_swr"] == pytest.approx(4.75)
    assert values["rate_baseline_upper_quartile_non_local"] == pytest.approx(9.5)
    assert values["rate_bin_count_swr"] == pytest.approx(0.0525)  # 3.5 per s * 15 ms
    # population a's rate over b's, within session: 1/2, 4/8, 8/1
    assert values["rate_baseline_ratio_swr_to_non_local"] == pytest.approx(0.5)
    assert values["rate_baseline_ratio_swr_coupled_to_no_swr"] == pytest.approx(0.5)
    assert values["rate_baseline_ratio_no_swr_to_swr"] == pytest.approx(8.0)
    assert values["rate_raw_immediate_session_non_local"] == pytest.approx(
        2.0 ** (1 / WEIGHTS[0, 0])
    )
    # the second reward coefficient is 0, the third log(lag-10 kernel) over its weight
    assert values["rate_bump_one_session_swr"] == 1.0
    assert values["rate_bump_two_session_swr"] == pytest.approx(
        1.1 ** (1 / WEIGHTS[2, -1])
    )
    assert values["rate_bump_two_animals_non_local_no_swr"] == 5
    assert values["rate_marginal_reward_child_better_fraction_swr"] == 1.0
    assert values["rate_hist_sessions_swr"] == 6
    assert values["rate_hist_separated_fraction_swr"] == 0.0
    for word, rung in (
        ("rew", "M1_rew"),
        ("histcos", "M3r_histcos"),
        ("time_histcos", "M6r_time_histcos"),
    ):
        assert values[f"rate_ladder_aic_{word}"] == AIC[rung]
    assert values["rate_ladder_sessions_reward_by_time"] == 6
    assert values["rate_arrival_time_sessions_swr"] == 6
    assert values["rate_arrival_time_cv_nan_fraction_non_local"] == 0.0
    # the in-transit record's own shares, and the all-stops record's counts
    assert values["rate_intransit_arrival_event_share_non_local"] == 0.3
    assert values["rate_intransit_arrival_event_share_low"] == 0.2
    assert values["rate_intransit_arrival_event_share_high"] == 0.5
    assert values["rate_allstops_arrival_bins_dropped"] == 250
    assert values["rate_allstops_arrival_bins_before"] == 1250
    assert values["rate_allstops_arrival_bins_dropped_fraction"] == 0.2
    assert values["rate_allstops_arrival_stops_dropped"] == 7
    assert values["rate_allstops_arrival_window_bins"] == 800
    assert values["rate_allstops_frozen_bins"] == 1000
    assert values["rate_allstops_arrival_window_fraction"] == 0.8


# --- the per-stop reward step ---------------------------------------------------------


def _per_stop_fits() -> pd.DataFrame:
    """R0 and R1 for each population in seven sessions, three of animal a.

    Session ``s`` (from 1) has ``AIC(R1) - AIC(R0)`` of ``s`` times the
    population's step. Every R0 fit and the first three R1 fits of a population
    carry a held-out score. The two rungs of the swr population's first session
    saw different stops.
    """
    steps = {
        "swr": -4.0,
        "non_local": -1.0,
        "non_local_swr": 2.0,
        "non_local_no_swr": 3.0,
    }
    rows = []
    for outcome, step in steps.items():
        for session, animal in enumerate(("a", "a", "a", "b", "c", "d", "e")):
            for model in ("R0", "R1"):
                rows.append(
                    {
                        "stop_cohort": "at_well",
                        "model": model,
                        "outcome": outcome,
                        "animal": animal,
                        "session_date": session,
                        "epoch": "02_r1",
                        "aic": 100.0 + (model == "R1") * step * (session + 1),
                        "n_stops": 11
                        if (outcome, model, session) == ("swr", "R0", 0)
                        else 10,
                        "cv_ll_per_stop_mean": -1.0
                        if model == "R0" or session < 3
                        else NAN,
                    }
                )
    return pd.DataFrame(rows)


def test_per_stop_values_are_the_child_minus_the_parent_on_the_same_stops():
    values = _values(behavior.per_stop_values(_per_stop_fits(), "source"))
    # non_local: steps -1 to -7; session median -4; animals a -2, then -4 to -7
    assert values["per_stop_sessions_non_local"] == 7
    assert values["per_stop_delta_aic_session_non_local"] == -4.0
    assert values["per_stop_delta_aic_animal_non_local"] == -5.0
    assert values["per_stop_child_better_fraction_non_local"] == 1.0
    assert values["per_stop_animals_child_better_non_local"] == 5
    # swr without its first session: steps -8 to -28; median -18; animal a -10
    assert values["per_stop_sessions_swr"] == 6
    assert values["per_stop_delta_aic_session_swr"] == -18.0
    assert values["per_stop_delta_aic_animal_swr"] == -20.0
    assert values["per_stop_delta_aic_session_non_local_no_swr"] == 12.0
    assert values["per_stop_child_better_fraction_non_local_swr"] == 0.0
    assert values["per_stop_animals_child_better_non_local_swr"] == 0
    # of the R1 fits, the first three sessions of each population carry a score
    assert values["per_stop_cv_finite_fraction_full"] == pytest.approx(3 / 7)


# --- the is_swr-contrast rate ladder --------------------------------------------------


def test_rate_contrast_separation_values_count_each_cohort_rung_and_term():
    term = "is_swr:reward_history_rcos_1"
    fits = pd.DataFrame(
        {
            # S1_rew carries no is_swr reward expansion; the last row is of
            # another population.
            "model": ["S1_rew"] * 4
            + ["S2_base"] * 3
            + ["S3_periswitch"] * 5
            + ["S2_base"],
            "population": ["contrast"] * 12 + ["swr"],
            "cohort": ["primary"] * 7 + ["post_outcome"] * 5 + ["primary"],
            f"coef_{term}": [NAN] * 4 + [0.1] * 9,
            "separated": [True, False, False, False]
            + [True, True, False]
            + [True, False, False, False, False]
            + [True],
            "separated_terms": ["reward_history_rcos_1"] * 2
            + [""] * 2
            + [term, "", ""]
            + [term] * 3
            + [""] * 2
            + [term],
        }
    )
    values = _values(behavior.rate_contrast_separation_values(fits, "source"))
    # contrast fits flagged separated: primary 1 of S1 and 2 of S2, of 7
    assert values["rate_contrast_separated_fraction_primary"] == pytest.approx(3 / 7)
    assert values["rate_contrast_separated_fraction_post_outcome"] == 0.2
    # the rungs carrying the term: S2 (3 fits, 2 separated, 1 flagging it) and
    # S3 (5 fits, 1 separated, 3 flagging it)
    assert values["rate_contrast_expanded_fits_low"] == 3
    assert values["rate_contrast_expanded_fits_high"] == 5
    assert values["rate_contrast_expanded_separated_low"] == 1
    assert values["rate_contrast_expanded_separated_high"] == 2
    assert values["rate_contrast_expanded_flagged_low"] == 1
    assert values["rate_contrast_expanded_flagged_high"] == 3
    assert values["rate_contrast_plain_fits"] == 4
    assert values["rate_contrast_plain_flagged"] == 2
    # of the nine fits of any population carrying the term, five flag it
    assert values["rate_contrast_term_separated_fraction"] == pytest.approx(5 / 9)


def test_rate_contrast_audit_values_name_the_contrast_term_that_loses_most():
    audited = pd.DataFrame(
        {
            "model": ["S2", "S3", "S4", "S9"],
            # the last row loses most, and is not a contrast row
            "population": ["contrast", "contrast", "contrast", "swr"],
            "term": ["t_two", "t_three", "t_four", "t_nine"],
            "n_sessions": [100, 90, 80, 10],
            "n_dropped": [0, 12, 5, 50],
        }
    )
    values = {v.key: v for v in behavior.rate_contrast_audit_values(audited, "source")}
    assert values["rate_contrast_audited_rows"].value == 4
    assert values["rate_contrast_terms_losing_sessions"].value == 2
    assert values["rate_contrast_most_dropped"].value == 12
    assert "S3 t_three" in values["rate_contrast_most_dropped"].description
    assert values["rate_contrast_most_dropped_of"].value == 102


# --- single coefficients: leaf ladder, choice, ripple power ---------------------------


def _coefficient_fits(term: str, coefficients, se: float) -> pd.DataFrame:
    """One fit per coefficient of ``term``, in the six sessions of five animals."""
    return pd.DataFrame(
        {
            "animal": list(ANIMALS),
            "session_date": list(range(len(ANIMALS))),
            "epoch": "02_r1",
            f"coef_{term}": coefficients,
            f"sd_{term}": 0.1,
            f"bse_{term}": se,
            "separated_terms": "",
        }
    )


def test_leaf_values_read_each_rung_s_own_term_and_step(tmp_path):
    rising = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # median 0.35
    _coefficient_fits(
        "reward_belief_variance_candidate_arm_preoutcome_z", rising, 0.05
    ).to_pickle(tmp_path / "Mu.pkl")
    _coefficient_fits(
        "reward_belief_mean_candidate_arm_preoutcome", [-2 * c for c in rising], 0.07
    ).to_pickle(tmp_path / "Mv.pkl")
    pd.DataFrame(
        {
            # the last row is another step into Mu, which the summary does not read
            "parent": ["M4", "M4", "Mu", "M3"],
            "child": ["Mu", "Mv", "Mvu", "Mu"],
            "n_animals": 5,
            "delta_aic_median": [1.0, 2.0, 3.0, 9.0],
            "child_better_aic": [10, 20, 30, 90],
            "n_sessions": [40, 50, 60, 90],
            "n_animals_child_better": [3, 4, 5, 1],
        }
    ).to_pickle(tmp_path / "model_comparison.pkl")
    values = _values(behavior.leaf_values(tmp_path, "source"))
    assert values["leaf_uncertainty"] == pytest.approx(0.35)
    assert values["leaf_value"] == pytest.approx(-0.7)
    assert values["leaf_uncertainty_sessions"] == 6
    assert values["leaf_uncertainty_animals_positive"] == 5
    assert values["leaf_value_animals_positive"] == 0
    assert values["leaf_uncertainty_se"] == pytest.approx(0.05)
    assert values["leaf_value_se"] == pytest.approx(0.07)
    assert values["leaf_step_uncertainty"] == 1.0
    assert values["leaf_step_value"] == 2.0
    assert values["leaf_step_both"] == 3.0
    assert values["leaf_step_value_clear"] == 20
    assert values["leaf_step_value_sessions"] == 50
    assert values["leaf_step_value_animals"] == 4


def test_choice_values_read_each_block_s_own_term():
    non_swr = [0.1, 0.2, 0.3, 0.4, 0.5, -0.6]  # median 0.25, five positive
    swr = [-0.1, -0.2, -0.3, -0.4, 0.5, 0.6]  # median -0.15, two positive
    split = _coefficient_fits("replay_fraction_non_swr", non_swr, 0.1).merge(
        _coefficient_fits("replay_fraction_swr", swr, 0.1)
    )
    fits = {
        "choice_swr_split": {"C_full_swr_split": split},
        "choice": {
            "C_full": _coefficient_fits("replay_fraction", [0.3] * 6, 0.1),
            "C_replay_only": _coefficient_fits("replay_fraction", [-0.9] * 6, 0.1),
        },
    }
    values = _values(behavior.choice_values(fits, "source"))
    assert values["choice_non_swr"] == pytest.approx(0.25)
    assert values["choice_swr"] == pytest.approx(-0.15)
    assert values["choice_full"] == pytest.approx(0.3)
    assert values["choice_raw"] == pytest.approx(-0.9)
    assert values["choice_non_swr_sessions"] == 6
    assert values["choice_non_swr_positive_sessions"] == 5
    assert values["choice_non_swr_negative_sessions"] == 1
    assert values["choice_swr_positive_sessions"] == 2
    assert values["choice_swr_negative_sessions"] == 4
    assert values["choice_full_animals_positive"] == 5
    assert values["choice_raw_animals_positive"] == 0
    # |coefficient| / 0.1 is 1 to 6: all but the first are above 1.96
    assert values["choice_non_swr_significant_fraction"] == pytest.approx(5 / 6)
    # six sessions of one sign: two-sided exact p = 2 / 2 ** 6
    assert values["choice_full_p"] == pytest.approx(0.03125)


def test_choice_values_read_through_a_refusing_gate_are_pending():
    # Twelve sessions. The SWR term is separated in three of them, a quarter,
    # which is more than the per-term gate lets be left out: the audited
    # aggregates would publish no row for it. Its seven values are kept, over
    # the nine sessions left, and marked pending; the other terms' are not.
    swr, non_swr = "replay_fraction_swr", "replay_fraction_non_swr"
    split = pd.DataFrame(
        {
            "animal": list(ANIMALS) * 2,
            "session_date": list(range(12)),
            "epoch": "02_r1",
            f"coef_{non_swr}": 0.2,
            f"sd_{non_swr}": 0.1,
            f"bse_{non_swr}": 0.1,
            f"coef_{swr}": -0.3,
            f"sd_{swr}": 0.1,
            f"bse_{swr}": 0.1,
            "separated_terms": [swr, swr, "", "", "", "", swr, "", "", "", "", ""],
        }
    )
    fits = {
        "choice_swr_split": {"C_full_swr_split": split},
        "choice": {
            "C_full": _coefficient_fits("replay_fraction", [0.3] * 6, 0.1),
            "C_replay_only": _coefficient_fits("replay_fraction", [-0.9] * 6, 0.1),
        },
    }

    values = {value.key: value for value in behavior.choice_values(fits, "source")}

    pending = {key for key, value in values.items() if value.pending is not None}
    assert pending == {
        "choice_swr",
        "choice_swr_p",
        "choice_swr_sessions",
        "choice_swr_positive_sessions",
        "choice_swr_negative_sessions",
        "choice_swr_animals_positive",
        "choice_swr_significant_fraction",
    }
    assert values["choice_swr_sessions"].value == 9
    assert values["choice_non_swr_sessions"].value == 12
    note = values["choice_swr"].pending
    assert "the per-term session gate refuses replay_fraction_swr" in note
    assert "more than 20 % of the sessions" in note


def test_power_values_read_each_covariate_s_own_column():
    rising = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    coefs = pd.DataFrame(
        {
            "animal": [*ANIMALS, "e"],
            "session_date": list(range(7)),
            "epoch": "02_r1",
            "known_reward_outcome_lag_1": rising,
            "rpe_postoutcome": [-c for c in rising],
            # one session has no coefficient for this covariate
            "patch_switch_fraction_last5_completed_trips": [2.0] * 6 + [NAN],
        }
    )
    values = _values(behavior.power_values(coefs, "source"))
    assert values["power_sessions"] == 7
    assert values["power_reward"] == pytest.approx(0.4)
    assert values["power_rpe"] == pytest.approx(-0.4)
    assert values["power_exploration"] == 2.0
    assert values["power_reward_animals_positive"] == 5
    assert values["power_rpe_animals_positive"] == 0
    assert values["power_reward_p"] == pytest.approx(2 / 2**7)
    assert values["power_exploration_p"] == pytest.approx(2 / 2**6)
