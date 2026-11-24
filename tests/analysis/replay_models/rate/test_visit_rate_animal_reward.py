"""One reward coefficient for each animal (``reward_level="animal"``).

The fits run on synthetic in-visit bins (``conftest.synthetic_visit_rows``)
whose counts are drawn from a planted rate (``visit_rate_scenarios``). The
reference is one joint penalized fit written here in NumPy: each session has
its own baseline and clock columns (the per-session design's), and the
sessions share one unpenalized reward column.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from continuum_swr_replay.analysis.regression.newton import CONVERGED, NO_FINITE_MAXIMUM
from continuum_swr_replay.analysis.regression.reml import assemble_penalty
from continuum_swr_replay.analysis.replay_models.rate import visit_rate as vr
from continuum_swr_replay.analysis.replay_models.rate import (
    visit_rate_scenarios as scenarios,
)
from continuum_swr_replay.analysis.replay_models.rate.visit_rate_statistics import (
    session_estimands,
    summarize_animal_rewards,
)

CLOCKS = scenarios.PlantedClocks(
    arrival=lambda s: 1.5 * (1.0 - np.exp(-np.asarray(s) / 1.5)),
    leaving=lambda u: -1.5 * np.exp(-np.asarray(u) / 0.3),
)
ANIMAL = vr.VisitRateSettings(reward_level="animal", include_stretch_clock=False)


def _animal_cohort(
    make_visit_rows,
    cohort_of,
    *,
    log_ratio: float,
    n_sessions: int,
    n_visits: int,
    total: float,
    seed: int,
    animal: str = "rat0",
    kind: str = "SWR",
) -> dict:
    """``n_sessions`` sessions of one animal, counts drawn at a constant
    planted log ratio."""
    rng = np.random.default_rng(seed)
    cohort = {}
    for i in range(n_sessions):
        key = (animal, f"2020010{i}", "02_r1")
        rows = make_visit_rows(key, n_visits=n_visits, seed=seed + i)
        truth = scenarios.constant_ratio(rows, CLOCKS, log_ratio=log_ratio, total=total)
        rows["count"] = rng.poisson(truth.expected).astype(float)
        cohort[key] = cohort_of(rows, kind)
    return cohort


def _joint_fit(design, cohort) -> tuple[float, float, list[np.ndarray]]:
    """The joint penalized Poisson fit by Newton's method: the sessions'
    blocks of the per-session design, block-diagonal, and one shared reward
    column, unpenalized. Returns the reward coefficient, its standard error
    (the inverse of the penalized Hessian) and each session's coefficients."""
    p = len(design.coef_names)
    blocks, ys, offsets = [], [], []
    for cohort_rows in cohort.values():
        rows = cohort_rows.rows["A"]
        built = design.build(rows)
        blocks.append((built.X, rows[vr.REWARD].to_numpy(dtype=float)))
        ys.append(rows[vr.RESPONSE].to_numpy(dtype=float))
        offsets.append(built.offset)
    n = len(blocks)
    X = np.zeros((sum(b[0].shape[0] for b in blocks), n * p + 1))
    start = 0
    for i, (Xi, r) in enumerate(blocks):
        X[start : start + len(r), i * p : (i + 1) * p] = Xi
        X[start : start + len(r), -1] = r
        start += len(r)
    y, offset = np.concatenate(ys), np.concatenate(offsets)
    S_one = assemble_penalty(design.blocks, design.log_penalties, p)
    S = np.zeros((n * p + 1, n * p + 1))
    for i in range(n):
        S[i * p : (i + 1) * p, i * p : (i + 1) * p] = S_one
    beta = np.zeros(n * p + 1)
    beta[[i * p for i in range(n)]] = np.log(y.sum() / np.exp(offset).sum())
    for _ in range(100):
        mu = np.exp(X @ beta + offset)
        H = (X.T * mu) @ X + S
        step = np.linalg.solve(H, X.T @ (y - mu) - S @ beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    mu = np.exp(X @ beta + offset)
    H = (X.T * mu) @ X + S
    se = float(np.sqrt(np.linalg.inv(H)[-1, -1]))
    return float(beta[-1]), se, [beta[i * p : (i + 1) * p] for i in range(n)]


def test_settings_of_the_animal_level() -> None:
    """At the animal level model A has no reward column in a session's
    design; the time-varying form and the per-session fit refuse it."""
    terms = vr.model_terms("A", "SWR", ANIMAL)
    assert not any(getattr(t, "column", None) == vr.REWARD for t in terms)
    assert vr.REWARD not in vr.tested_terms("A", ANIMAL)
    with pytest.raises(ValueError, match="reward_level"):
        vr.VisitRateSettings(reward_level="trial")
    with pytest.raises(ValueError, match="time_varying"):
        vr.VisitRateSettings(reward_level="animal", reward_form="time_varying")


def test_equals_one_joint_fit(make_visit_rows, cohort_of) -> None:
    """Three sessions of one animal: the coefficient, its standard error and
    each session's coefficients equal the joint fit's to 1e-6; Q1 and Q7a are
    read from the records, Q2 is not a session value."""
    cohort = _animal_cohort(
        make_visit_rows,
        cohort_of,
        log_ratio=0.6,
        n_sessions=3,
        n_visits=80,
        total=400.0,
        seed=3,
    )
    fit = vr.fit_animal_rewards(
        cohort, kind="SWR", settings=ANIMAL, allow_provisional="test"
    )
    (animal,) = fit.animals
    assert animal.status == CONVERGED and animal.n_iter <= vr.ANIMAL_REWARD_MAX_ITER
    beta, se, coefs = _joint_fit(fit.design, cohort)
    assert animal.coefficient == pytest.approx(beta, abs=1e-6)
    assert animal.se == pytest.approx(se, rel=1e-6)
    for record, coef in zip(animal.records, coefs, strict=True):
        np.testing.assert_allclose(record.coef, coef, atol=1e-6)
    assert animal.n_sessions == 3
    assert animal.n_events == sum(
        int(c.rows["A"]["count"].sum()) for c in cohort.values()
    )
    contrasts = vr.declared_contrasts(fit.design, ANIMAL)
    assert set(contrasts) == {"Q1", "Q7a"}
    values = session_estimands(fit.session_fit())
    assert set(values["estimand_id"]) == {"Q1", "Q7a"}
    assert np.isfinite(values["estimate"]).all()


def test_a_planted_coefficient_is_recovered(make_visit_rows, cohort_of) -> None:
    """Six sessions of one animal with a planted log ratio of 0.7 and about
    1,000 events each: the coefficient is within 3 standard errors of it
    (a simulation tolerance)."""
    cohort = _animal_cohort(
        make_visit_rows,
        cohort_of,
        log_ratio=0.7,
        n_sessions=6,
        n_visits=100,
        total=1000.0,
        seed=11,
    )
    (animal,) = vr.fit_animal_rewards(
        cohort, kind="SWR", settings=ANIMAL, allow_provisional="test"
    ).animals
    assert abs(animal.coefficient - 0.7) < 3 * animal.se
    assert animal.se < 0.1


def test_a_session_without_an_event_on_one_side_takes_part(
    make_visit_rows, cohort_of
) -> None:
    """A session whose bins after an omission hold no event is fitted with
    the others: no rule reads its events, and it enters the coefficient."""
    cohort = _animal_cohort(
        make_visit_rows,
        cohort_of,
        log_ratio=0.5,
        n_sessions=3,
        n_visits=60,
        total=300.0,
        seed=21,
    )
    key = next(iter(cohort))
    rows = cohort[key].rows["A"].copy()
    rows.loc[rows[vr.REWARD] == 0.0, vr.RESPONSE] = 0.0
    cohort[key] = cohort_of(rows, "SWR")
    fit = vr.fit_animal_rewards(
        cohort, kind="SWR", settings=ANIMAL, allow_provisional="test"
    )
    (animal,) = fit.animals
    assert animal.status == CONVERGED and animal.n_sessions == 3
    assert all(r.status == CONVERGED and not r.no_fit for r in animal.records)
    beta, _, _ = _joint_fit(fit.design, cohort)
    assert animal.coefficient == pytest.approx(beta, abs=1e-6)


def test_an_animal_without_rows_on_one_side_is_refused(
    make_visit_rows, cohort_of
) -> None:
    """Every bin of the animal follows a reward: its coefficient is not
    identified, and the error names the animal and the side."""
    cohort = _animal_cohort(
        make_visit_rows,
        cohort_of,
        log_ratio=0.5,
        n_sessions=2,
        n_visits=40,
        total=200.0,
        seed=31,
    )
    cohort = {
        key: cohort_of(c.rows["A"].assign(**{vr.REWARD: 1.0}), "SWR")
        for key, c in cohort.items()
    }
    with pytest.raises(ValueError, match="rat0.*after an omission"):
        vr.fit_animal_rewards(
            cohort, kind="SWR", settings=ANIMAL, allow_provisional="test"
        )


def test_the_animal_level_refuses_provisional_penalties_in_a_real_fit(
    make_visit_rows, cohort_of
) -> None:
    cohort = _animal_cohort(
        make_visit_rows,
        cohort_of,
        log_ratio=0.5,
        n_sessions=2,
        n_visits=40,
        total=200.0,
        seed=41,
    )
    with pytest.raises(ValueError, match="provisional"):
        vr.fit_animal_rewards(cohort, kind="SWR", settings=ANIMAL)
    with pytest.raises(ValueError, match="fit_animal_rewards"):
        vr.fit_visit_rate(
            cohort, model="A", kind="SWR", settings=ANIMAL, allow_provisional="test"
        )
    session = dataclasses.replace(ANIMAL, reward_level="session")
    with pytest.raises(ValueError, match="reward_level"):
        vr.fit_animal_rewards(
            cohort, kind="SWR", settings=session, allow_provisional="test"
        )


def test_the_cohort_value_is_the_mean_over_animals_with_its_t_interval() -> None:
    """Per kind, the mean of the animals' coefficients with the t interval
    over the animals; the difference SWR minus No SWR taken animal by animal,
    over the animals with both."""
    values = pd.DataFrame(
        {
            "animal": ["a", "b", "c", "d", "a", "b", "c"],
            "kind": ["SWR"] * 4 + ["No SWR"] * 3,
            "coefficient": [1.0, 0.8, 0.6, np.nan, 0.1, 0.3, -0.1],
        }
    )
    table = summarize_animal_rewards(values).set_index("tested")
    t3 = scipy.stats.t.ppf(0.975, 2)
    swr = np.array([1.0, 0.8, 0.6])
    assert table.loc["SWR", "estimate"] == pytest.approx(swr.mean())
    assert table.loc["SWR", "n_animals"] == 3
    half = t3 * swr.std(ddof=1) / np.sqrt(3)
    assert table.loc["SWR", "high"] == pytest.approx(swr.mean() + half)
    difference = swr - np.array([0.1, 0.3, -0.1])
    assert table.loc["SWR minus No SWR", "estimate"] == pytest.approx(difference.mean())
    assert table.loc["SWR minus No SWR", "low"] == pytest.approx(
        difference.mean() - t3 * difference.std(ddof=1) / np.sqrt(3)
    )
    assert table.loc["No SWR", "n_animals_same_sign"] == 2


def test_an_animal_without_an_event_on_one_side_has_no_finite_coefficient(
    make_visit_rows, cohort_of
) -> None:
    """No session of rat1 holds an event after an omission: the joint
    likelihood rises without bound, so rat1 has no coefficient (not fitted,
    ``no finite maximum``), and rat0 is fitted as before."""
    first = _animal_cohort(
        make_visit_rows,
        cohort_of,
        log_ratio=0.5,
        n_sessions=2,
        n_visits=40,
        total=200.0,
        seed=51,
    )
    second = _animal_cohort(
        make_visit_rows,
        cohort_of,
        log_ratio=0.5,
        n_sessions=2,
        n_visits=40,
        total=200.0,
        seed=61,
        animal="rat1",
    )
    for key, c in second.items():
        rows = c.rows["A"].copy()
        rows.loc[rows[vr.REWARD] == 0.0, vr.RESPONSE] = 0.0
        second[key] = cohort_of(rows, "SWR")
    fit = vr.fit_animal_rewards(
        {**first, **second}, kind="SWR", settings=ANIMAL, allow_provisional="test"
    )
    rat0, rat1 = fit.animals
    assert rat0.status == CONVERGED and np.isfinite(rat0.coefficient)
    assert rat1.status == NO_FINITE_MAXIMUM and np.isnan(rat1.coefficient)
    assert rat1.records == [] and rat1.n_sessions == 0
