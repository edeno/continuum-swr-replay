"""The visit rate model's planted scenarios (``rate/visit_rate_scenarios.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.rate import visit_rate as vr
from continuum_swr_replay.analysis.replay_models.rate import (
    visit_rate_scenarios as scenarios,
)

KEY = ("rat", "20200101", "02_r1")
#: Linear curves: Q1 = 0.4 * (3 - 0.5) and Q7a = -2 * (0.125 - 1.5).
LINEAR = scenarios.PlantedClocks(
    arrival=lambda s: 0.4 * np.asarray(s), leaving=lambda u: -2.0 * np.asarray(u)
)


@pytest.fixture
def rows(make_visit_rows) -> pd.DataFrame:
    return make_visit_rows(KEY, n_visits=40, seed=3, length_s=(1.5, 12.0))


def test_planted_curve_values_are_the_hand_values(rows) -> None:
    planted = scenarios.no_effect(rows, LINEAR)
    assert planted.values == pytest.approx({"Q1": 1.0, "Q2": 0.0, "Q7a": 2.75})
    later = scenarios.no_effect(rows, LINEAR, q1_reference_s=(1.0, 2.0))
    assert later.values["Q1"] == pytest.approx(0.4 * (3.0 - 1.5))
    s, u = rows[vr.ARRIVAL].to_numpy(), rows[vr.LEAVING].to_numpy()
    expected = rows[vr.EXPOSURE].to_numpy() * np.exp(0.4 * s - 2.0 * u)
    assert planted.expected == pytest.approx(expected)


def test_a_constant_ratio_and_the_session_total(rows) -> None:
    planted = scenarios.constant_ratio(rows, LINEAR, log_ratio=0.5, total=250.0)
    base = scenarios.no_effect(rows, LINEAR, total=250.0)
    assert planted.expected.sum() == pytest.approx(250.0)
    assert planted.values["Q2"] == 0.5
    ratio = planted.expected / base.expected
    rewarded = rows[vr.REWARD].to_numpy() == 1.0
    assert np.ptp(ratio[rewarded]) == pytest.approx(0.0, abs=1e-12)
    assert ratio[rewarded][0] / ratio[~rewarded][0] == pytest.approx(np.exp(0.5))


def test_a_stretch_curve_enters_the_rate(rows) -> None:
    clocks = scenarios.PlantedClocks(
        LINEAR.arrival, LINEAR.leaving, stretch=lambda m: -np.asarray(m)
    )
    with_m = scenarios.no_effect(rows, clocks).expected
    without = scenarios.no_effect(rows, LINEAR).expected
    assert with_m / without == pytest.approx(np.exp(-rows[vr.STRETCH].to_numpy()))


def test_a_fading_ratio(rows) -> None:
    """``g`` is the full log ratio before the fade, 0 after it; planted Q2
    lies between, and is the full ratio when no bin reaches the fade."""
    planted = scenarios.fading_ratio(
        rows, LINEAR, log_ratio=0.8, fade_start_s=3.0, fade_end_s=6.0
    )
    base = scenarios.no_effect(rows, LINEAR).expected
    s = rows[vr.ARRIVAL].to_numpy()
    rewarded = rows[vr.REWARD].to_numpy() == 1.0
    g = np.log(planted.expected / base)
    assert g[rewarded & (s <= 3.0)] == pytest.approx(0.8)
    assert g[rewarded & (s >= 6.0)] == pytest.approx(0.0, abs=1e-12)
    assert g[~rewarded] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < planted.values["Q2"] < 0.8
    mu0 = base[rewarded]
    assert planted.values["Q2"] == pytest.approx(
        np.log(np.sum(mu0 * np.exp(g[rewarded])) / mu0.sum())
    )
    never = scenarios.fading_ratio(
        rows, LINEAR, log_ratio=0.8, fade_start_s=50.0, fade_end_s=60.0
    )
    assert never.values["Q2"] == pytest.approx(0.8)


def test_a_ratio_that_differs_between_sessions_and_animals(make_visit_rows) -> None:
    sessions = pd.DataFrame(
        [(f"rat{a}", f"2020010{d}", "02_r1") for a in range(5) for d in range(4)],
        columns=["animal", "session_date", "epoch"],
    )
    ratios = scenarios.session_log_ratios(
        sessions, mean=0.9, animal_sd=0.2, session_sd=0.3, seed=7
    )
    assert ratios == scenarios.session_log_ratios(
        sessions, mean=0.9, animal_sd=0.2, session_sd=0.3, seed=7
    )
    assert len(ratios) == 20 and len(set(ratios.values())) == 20
    key = ("rat2", "20200103", "02_r1")
    rows = make_visit_rows(key, n_visits=20, seed=1)
    planted = scenarios.differing_ratio(rows, LINEAR, log_ratios=ratios)
    assert planted.values["Q2"] == ratios[key]
    constant = scenarios.constant_ratio(rows, LINEAR, log_ratio=ratios[key])
    assert planted.expected == pytest.approx(constant.expected)


def test_a_later_onset_after_an_omission(rows) -> None:
    planted = scenarios.later_onset_after_omission(rows, LINEAR, delay_s=0.5)
    base = scenarios.no_effect(rows, LINEAR).expected
    rewarded = rows[vr.REWARD].to_numpy() == 1.0
    s = rows[vr.ARRIVAL].to_numpy()
    assert planted.expected[rewarded] == pytest.approx(base[rewarded])
    shift = 0.4 * (np.maximum(s - 0.5, 0.0) - s)
    assert planted.expected[~rewarded] == pytest.approx(
        base[~rewarded] * np.exp(shift[~rewarded])
    )
    assert np.isnan(planted.values["Q2"])
    assert planted.values["Q1"] == pytest.approx(1.0)


def test_a_rate_table_truth(rows) -> None:
    """A table whose rate after a reward is twice that after an omission and
    rises tenfold after 2 s since arrival: planted Q2 is log 2, Q1 log 10 and
    Q7a 0 (the table is flat in time until leaving); the table made back from
    expected counts on the same bins is the table."""
    arrival_edges = np.array([0.0, 2.0, np.inf])
    leaving_edges = np.array([0.0, 1.0, np.inf])
    rates = np.empty((2, 2, 2))
    rates[0] = [[0.1, 0.1], [1.0, 1.0]]
    rates[1] = 2.0 * rates[0]
    table = scenarios.RateTable(rates, arrival_edges, leaving_edges)
    planted = scenarios.rate_table_truth(rows, table)
    assert planted.values["Q2"] == pytest.approx(np.log(2.0))
    assert planted.values["Q7a"] == pytest.approx(0.0, abs=1e-12)
    # The windows are closed: a bin centered on 2 s would sit in both, and
    # none does here, so Q1 is the ratio of the two cells' rates.
    assert not np.isclose(rows[vr.ARRIVAL], 2.0).any()
    assert planted.values["Q1"] == pytest.approx(np.log(10.0))
    scaled = scenarios.rate_table_truth(rows, table, total=100.0)
    assert scaled.expected.sum() == pytest.approx(100.0)
    rebuilt = scenarios.rate_table_from_bins(
        rows.assign(count=planted.expected),
        arrival_edges,
        leaving_edges,
        shrink_s=0.0,
        floor=0.0,
    )
    assert rebuilt.rates == pytest.approx(rates)
    floored = scenarios.rate_table_from_bins(
        rows, arrival_edges, leaving_edges, shrink_s=50.0, floor=5e-4
    )
    assert (floored.rates == 5e-4).all()
