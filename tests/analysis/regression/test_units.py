"""The reduction of one estimate per unit of a fit (``regression/units.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis import shared_statistics as stats
from continuum_swr_replay.analysis.regression import units

NAN = np.nan


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
    got = units.unit_summary(_SESSION_VALUES, "session", n_draws=300, seed=4)
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
    got = units.unit_summary(days, "day", n_draws=300, seed=4)
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
    got = units.unit_summary(values, "animal", n_draws=300, seed=4)
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
        set(units.unit_summary(values, fit_by, n_draws=50, seed=1))
        for values, fit_by in ((session, "session"), (day, "day"), (animal, "animal"))
    ]
    assert keys[0] == keys[1] == keys[2]


def test_unit_summary_refuses_values_not_keyed_by_the_unit():
    with pytest.raises(ValueError, match="day"):
        units.unit_summary(_SESSION_VALUES, "day", n_draws=50, seed=1)
    with pytest.raises(ValueError, match="fit_by"):
        units.unit_summary(_SESSION_VALUES, "epoch", n_draws=50, seed=1)
    with pytest.raises(ValueError, match="more than one value"):
        units.unit_summary(
            pd.Series([1.0, 2.0], index=pd.Index(["a", "a"], name="animal")),
            "animal",
            n_draws=50,
            seed=1,
        )


def test_same_sign_counted_by_one_rule_at_every_unit():
    # Five animals, one session and one day each: 0, 1e-17, 1, 2, -1. The mean is
    # 0.4. Read to 12 significant digits of the largest value (2), 1e-17 is zero,
    # so neither it nor the exact zero has a sign: two animals share the mean's.
    values = [0.0, 1e-17, 1.0, 2.0, -1.0]
    animals = list("abcde")
    session = pd.Series(
        values,
        index=pd.MultiIndex.from_tuples([(a, "d1", "e1") for a in animals]),
    )
    day = pd.Series(
        values, index=pd.MultiIndex.from_tuples([(a, "d1") for a in animals])
    )
    animal = pd.Series(values, index=pd.Index(animals))
    counts = [
        units.unit_summary(v, fit_by, n_draws=50, seed=1)["n_animals_same_sign"]
        for v, fit_by in ((session, "session"), (day, "day"), (animal, "animal"))
    ]
    assert counts == [2, 2, 2]
    assert stats.animal_mean_interval(animal)["n_animals_same_sign"] == 2


def test_unit_values_and_estimate_of_records() -> None:
    """The records of a fit at each unit give their values indexed by the
    unit key (``FitRecord.unit_key``); the estimate alone equals the
    summary's."""
    from continuum_swr_replay.analysis.regression.likelihoods import Poisson
    from continuum_swr_replay.analysis.regression.sessions import fit_sessions
    from continuum_swr_replay.analysis.regression.terms import (
        Offset,
        Slope,
        build_design,
    )

    from .conftest import rate_rows

    sessions = {}
    for a in range(3):
        for d in range(2):
            key = (f"a{a}", f"2026000{d + 1}", "02_r1")
            sessions[key] = rate_rows(10 * a + d, n_visits=30)
    design = build_design(
        [Slope("last_rewarded", 0.0), Offset("exposure")],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=True,
        baseline=("(Intercept)",),
    )
    for fit_by, n in (("session", 6), ("day", 6), ("animal", 3)):
        records = fit_sessions(
            design, sessions, Poisson(), model="m", kind="SWR", response="count",
            unit="well_visit_id", table_sha256={}, fit_by=fit_by,
        )  # fmt: skip
        values = units.unit_values(np.arange(n, dtype=float), records)
        assert list(values.index) == [r.unit_key for r in records]
        assert values.index.nlevels == {"session": 3, "day": 2, "animal": 1}[fit_by]
        summary = units.unit_summary(values, fit_by, n_draws=100, seed=3)
        assert units.unit_estimate(values, fit_by) == summary["estimate"]
    assert records[0].unit_key == ("a0",)
    assert np.isnan(units.unit_estimate(values * np.nan, "animal"))
