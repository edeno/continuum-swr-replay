"""Part 4's stop-context values from a synthetic results table."""

import itertools

import pandas as pd
import pytest

from continuum_swr_replay.reported.part4 import _value_factory, stop_context_values


def _rows(n_loao_estimated: int = 5) -> list[dict]:
    rows = []
    for event_class, sign, unit_set, estimator, term in itertools.product(
        ("non_local_SWR", "non_local_no_SWR"),
        ("increase", "decrease"),
        ("called", "all"),
        ("conditional", "plugin"),
        ("rewarded", "next_trip_switches_patch", "overlap"),
    ):
        rows.append(
            {
                "event_class": event_class,
                "sign": sign,
                "unit_set": unit_set,
                "estimator": estimator,
                "term": term,
                "exp_coef": 1.08 if estimator == "conditional" else 1.10,
                "ci_low": 1.02,
                "ci_high": 1.14,
                "n_animals_pooled_sign": 3,
                "n_animals_estimated": 5,
                "loao_exp_coef_min": 1.02,
                "loao_exp_coef_max": 1.10,
                "n_animals": 5,
                "n_loao_estimated": n_loao_estimated,
            }
        )
    return rows


def test_stop_context_values_main_and_check_rows():
    make = _value_factory("synthetic", ("20",), ("paper",))

    values = {v.key: v.value for v in stop_context_values(pd.DataFrame(_rows()), make)}

    assert values["stop_context_reward_gain_swr_replay_increase"] == 1.08
    assert (
        values["stop_context_reward_gain_swr_replay_increase_animals_pooled_sign"] == 3
    )
    assert "stop_context_reward_gain_non_swr_replay_increase_all_units" in values
    # Only the reward check for increases is given over all units; no overlap.
    assert "stop_context_switch_gain_swr_replay_increase_all_units" not in values
    assert not any("overlap" in key for key in values)
    assert all(not any(ch.isdigit() for ch in key) for key in values)


def test_a_failed_leave_one_out_fit_is_refused():
    make = _value_factory("synthetic", ("20",), ("paper",))
    with pytest.raises(ValueError, match="4 of 5 leave-one-animal-out fits"):
        stop_context_values(pd.DataFrame(_rows(n_loao_estimated=4)), make)
