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


COEF = vr.SECONDARY_ESTIMAND


def _windows(value: float) -> dict[str, float]:
    return {"Q2": value, "Q2-onset": value}


def test_planted_curve_values_are_the_hand_values(rows) -> None:
    planted = scenarios.no_effect(rows, LINEAR)
    assert planted.by_outcome == pytest.approx(
        {"Q1": 1.0, "Q7a": 2.75, **_windows(0.0)}, abs=1e-9
    )
    assert planted.pooled == pytest.approx({COEF: 0.0}, abs=1e-9)
    assert planted.for_form("coefficient") is planted.pooled
    assert planted.for_form("by_outcome") is planted.by_outcome
    with pytest.raises(ValueError, match="time_varying"):
        planted.for_form("time_varying")
    later = scenarios.no_effect(rows, LINEAR, q1_reference_s=(1.0, 2.0))
    assert later.by_outcome["Q1"] == pytest.approx(0.4 * (3.0 - 1.5))
    s, u = rows[vr.ARRIVAL].to_numpy(), rows[vr.LEAVING].to_numpy()
    expected = rows[vr.EXPOSURE].to_numpy() * np.exp(0.4 * s - 2.0 * u)
    assert planted.expected == pytest.approx(expected)


def test_a_constant_ratio_and_the_session_total(rows) -> None:
    planted = scenarios.constant_ratio(rows, LINEAR, log_ratio=0.5, total=250.0)
    base = scenarios.no_effect(rows, LINEAR, total=250.0)
    assert planted.expected.sum() == pytest.approx(250.0)
    # A constant effect is the same in every window and is the coefficient.
    assert planted.by_outcome == pytest.approx(
        {"Q1": 1.0, "Q7a": 2.75, **_windows(0.5)}, abs=1e-9
    )
    assert planted.pooled == pytest.approx({COEF: 0.5}, abs=1e-9)
    ratio = planted.expected / base.expected
    rewarded = rows[vr.REWARD].to_numpy() == 1.0
    assert np.ptp(ratio[rewarded]) == pytest.approx(0.0, abs=1e-12)
    assert ratio[rewarded][0] / ratio[~rewarded][0] == pytest.approx(np.exp(0.5))


def test_a_fading_ratio(rows) -> None:
    """``g`` is the full log ratio before the fade, 0 after it. The window
    truths are the plain means of ``g`` over each window's grid (the fade
    from 3 to 6 s: 0.8 * mean(clip((6 - s) / 3, 0, 1))); Q1 and Q7a are the
    omission side's curves'. The coefficient's projection lies between 0 and
    the full ratio; with no bin in the fade every truth is the full ratio."""
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
    for name, (lo, hi) in (("Q2", (4.0, 10.0)), ("Q2-onset", (2.0, 4.0))):
        grid = np.linspace(lo, hi, round((hi - lo) * 100) + 1)
        hand = 0.8 * np.mean(np.clip((6.0 - grid) / 3.0, 0.0, 1.0))
        assert planted.by_outcome[name] == pytest.approx(hand)
    assert planted.by_outcome["Q2"] < planted.by_outcome["Q2-onset"] < 0.8
    assert planted.by_outcome["Q1"] == pytest.approx(1.0)
    assert planted.by_outcome["Q7a"] == pytest.approx(2.75)
    assert 0.0 < planted.pooled[COEF] < 0.8
    never = scenarios.fading_ratio(
        rows, LINEAR, log_ratio=0.8, fade_start_s=50.0, fade_end_s=60.0
    )
    assert never.by_outcome == pytest.approx(
        {"Q1": 1.0, "Q7a": 2.75, **_windows(0.8)}, abs=1e-9
    )
    assert never.pooled == pytest.approx({COEF: 0.8}, abs=1e-9)


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
    assert planted.by_outcome["Q2"] == pytest.approx(ratios[key])
    assert planted.pooled[COEF] == pytest.approx(ratios[key])
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
    # No ratio is planted: the coefficient has no truth. The window truths
    # are the means of g(s) = 0.4 * (s - max(s - 0.5, 0)), 0.2 from 0.5 s on.
    assert np.isnan(planted.pooled[COEF])
    assert planted.by_outcome["Q2"] == pytest.approx(0.2)
    assert planted.by_outcome["Q2-onset"] == pytest.approx(0.2)
    # The omission side's arrival curve is the delayed one: 0.4 * max(s -
    # 0.5, 0) has mean 0.4 * 2.5 over [2, 4] and 0.4 * 0.125 over [0, 1]
    # (the line from 0.5 to 1, half the window), on the 0.01 s grid.
    grid = np.linspace(0.0, 1.0, 101)
    reference = 0.4 * np.mean(np.maximum(grid - 0.5, 0.0))
    assert planted.by_outcome["Q1"] == pytest.approx(0.4 * 2.5 - reference)
    assert planted.by_outcome["Q7a"] == pytest.approx(2.75)


def test_the_drop_before_leaving(rows) -> None:
    """A drop on the leaving clock only: Q7a is the drop (both windows of
    Q7a on either side of the step); no reward effect is planted."""
    leaving = scenarios.drop_before_leaving(rows, LINEAR.arrival, drop=1.2, width_s=0.3)
    u = rows[vr.LEAVING].to_numpy()
    s = rows[vr.ARRIVAL].to_numpy()
    exposure = rows[vr.EXPOSURE].to_numpy()
    assert leaving.expected == pytest.approx(
        exposure * np.exp(0.4 * s - 1.2 * (u < 0.3))
    )
    assert leaving.by_outcome == pytest.approx(
        {"Q1": 1.0, "Q7a": -1.2, **_windows(0.0)}, abs=1e-9
    )
    assert leaving.pooled == pytest.approx({COEF: 0.0}, abs=1e-9)


def _poisson_steps(
    rows: pd.DataFrame,
    expected: np.ndarray,
    *,
    by_outcome: bool,
    arrival_cell_s: float,
    leaving_cell_s: float,
) -> tuple[dict, dict, float]:
    """An independent maximum-likelihood fit by Newton's method of
    ``expected`` on indicators of the cells (``floor(clock / width)``, the
    clocks clamped at 20 and 3 s), the reward, the intercept and the offset
    ``log(exposure)``: each arrival cell's step (of each outcome when
    ``by_outcome``, keyed ``(outcome, cell)``, else ``(0, cell)``), each
    leaving cell's step and the reward coefficient."""
    reward = rows[vr.REWARD].to_numpy().astype(int)
    a_cell = np.floor(np.minimum(rows[vr.ARRIVAL].to_numpy(), 20.0) / arrival_cell_s)
    u_cell = np.floor(np.minimum(rows[vr.LEAVING].to_numpy(), 3.0) / leaving_cell_s)
    a_key = list(
        zip(reward if by_outcome else 0 * reward, a_cell.astype(int), strict=True)
    )
    a_levels = sorted(set(a_key))
    u_levels = sorted(set(u_cell.astype(int)))
    columns = [np.ones(len(rows))]
    columns += [np.array([k == level for k in a_key], float) for level in a_levels[1:]]
    columns += [(u_cell == level).astype(float) for level in u_levels[1:]]
    if not by_outcome:
        columns.append(reward.astype(float))
    X = np.column_stack(columns)
    offset = np.log(rows[vr.EXPOSURE].to_numpy())
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(expected.sum() / np.exp(offset).sum())
    for _ in range(100):
        mu = np.exp(offset + X @ beta)
        step = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (expected - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    n_a = len(a_levels) - 1
    arrival = dict(zip(a_levels, [0.0, *beta[1 : 1 + n_a]], strict=True))
    leaving = dict(
        zip(u_levels, [0.0, *beta[1 + n_a : len(u_levels) + n_a]], strict=True)
    )
    return arrival, leaving, (np.nan if by_outcome else float(beta[-1]))


def test_a_rate_table_truth(make_visit_rows) -> None:
    """A table whose rates change with both clocks and the outcome, with
    cells that the windows of Q1 and Q7a straddle: both truths equal an
    independent Poisson fit of the planted expected counts on cell
    indicators, read at the windows' grid points."""
    rows = make_visit_rows(KEY, n_visits=80, seed=4, length_s=(1.5, 6.0))
    arrival_edges = np.array([0.0, 0.6, 2.5, 3.5, np.inf])
    leaving_edges = np.array([0.0, 0.2, 1.2, np.inf])
    rates = np.random.default_rng(0).uniform(0.2, 3.0, (2, 4, 3))
    table = scenarios.RateTable(rates, arrival_edges, leaving_edges)
    cells = {"arrival_cell_s": 0.5, "leaving_cell_s": 0.1}
    planted = scenarios.rate_table_truth(rows, table, **cells)
    outcome, a, u = table.cells(rows)
    expected = rows[vr.EXPOSURE].to_numpy() * rates[outcome, a, u]
    assert planted.expected == pytest.approx(expected)

    def grid_mean(steps: dict, outcome: int, lo: float, hi: float, width: float):
        grid = np.arange(round(lo * 100), round(hi * 100) + 1) / 100.0
        return np.mean([steps[(outcome, int(np.floor(x / width)))] for x in grid])

    arrival, leaving, _ = _poisson_steps(rows, expected, by_outcome=True, **cells)
    q1 = grid_mean(arrival, 0, 2.0, 4.0, 0.5) - grid_mean(arrival, 0, 0.0, 1.0, 0.5)
    steps_u = {(0, k): v for k, v in leaving.items()}
    q7a = grid_mean(steps_u, 0, 0.0, 0.25, 0.1) - grid_mean(steps_u, 0, 1.0, 2.0, 0.1)
    difference = {
        (1, c): arrival.get((1, c), np.nan) - arrival.get((0, c), np.nan)
        for c in range(41)
    }
    windows = {
        "Q2": grid_mean(difference, 1, 4.0, 10.0, 0.5),
        "Q2-onset": grid_mean(difference, 1, 2.0, 4.0, 0.5),
    }
    assert planted.by_outcome == pytest.approx(
        {"Q1": q1, "Q7a": q7a, **windows}, abs=1e-8, nan_ok=True
    )
    *_, beta = _poisson_steps(rows, expected, by_outcome=False, **cells)
    assert planted.pooled == pytest.approx({COEF: beta}, abs=1e-8)
    scaled = scenarios.rate_table_truth(rows, table, total=100.0, **cells)
    assert scaled.expected.sum() == pytest.approx(100.0)
    assert scaled.pooled == pytest.approx(planted.pooled, abs=1e-8)
    rebuilt = scenarios.rate_table_from_bins(
        rows.assign(count=planted.expected),
        arrival_edges,
        leaving_edges,
        shrink_s=1e-12,
        floor=0.0,
    )
    occupied = np.zeros(rates.shape, dtype=bool)
    occupied[outcome, a, u] = True
    assert rebuilt.rates[occupied] == pytest.approx(rates[occupied])
    floored = scenarios.rate_table_from_bins(
        rows, arrival_edges, leaving_edges, shrink_s=50.0, floor=5e-4
    )
    assert (floored.rates == 5e-4).all()


def test_a_drop_before_small_movements(rows) -> None:
    """A drop on the time until the end of the immobile stretch only (a
    column no model reads): the rate drops in the last 0.3 s of every
    stretch, the leaving curve is flat, so the planted Q7a is 0, Q1 the
    arrival curve's, and no reward effect is planted. It checks the
    two-clock model on data it cannot represent."""
    stretch = scenarios.drop_before_small_movements(
        rows, LINEAR.arrival, drop=1.2, width_s=0.3
    )
    s = rows[vr.ARRIVAL].to_numpy()
    m = rows[scenarios.STRETCH_COLUMN].to_numpy()
    exposure = rows[vr.EXPOSURE].to_numpy()
    assert stretch.expected == pytest.approx(
        exposure * np.exp(0.4 * s - 1.2 * (m < 0.3))
    )
    assert stretch.by_outcome == pytest.approx(
        {"Q1": 1.0, "Q7a": 0.0, **_windows(0.0)}, abs=1e-9
    )
    assert stretch.pooled == pytest.approx({COEF: 0.0}, abs=1e-9)
