"""The declared family of tests: twelve estimands, each for SWR, No SWR and their difference."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.regression import declared_tests as dt
from continuum_swr_replay.analysis.regression.contrasts import (
    range_grid,
    range_mean_contrast,
)

SETTLED_IDS = {
    "Q1",
    "Q2",
    "Q3a",
    "Q3b",
    "Q4a",
    "Q4b",
    "Q5a",
    "Q5b-var",
    "Q5b-n",
    "Q6",
    "Q7a",
    "Q7b",
}


def test_list_is_unique_and_complete() -> None:
    ids = [e.estimand_id for e in dt.ESTIMANDS]
    assert len(ids) == len(set(ids))
    assert set(ids) == SETTLED_IDS
    for e in dt.ESTIMANDS:
        assert e.question and e.description and e.model
        assert e.scale in ("log rate ratio", "log share ratio")
        assert isinstance(
            e.contrast, dt.CurveContrast | dt.SDChangeContrast | dt.CoefficientContrast
        )
    # Each estimand is tested for SWR, for No SWR and for their difference.
    assert dt.TESTED == ("SWR", "No SWR", "SWR minus No SWR")
    assert len(dt.DECLARED_TESTS) == len(dt.ESTIMANDS) * len(dt.TESTED)
    assert len(set(dt.DECLARED_TESTS)) == len(dt.DECLARED_TESTS)
    for e in dt.ESTIMANDS:
        assert {kind for i, kind in dt.DECLARED_TESTS if i == e.estimand_id} == set(
            dt.TESTED
        )


def test_estimand_looks_up_by_id() -> None:
    assert dt.estimand("Q2").contrast == dt.CoefficientContrast(
        plus=("last_outcome_rewarded",)
    )
    with pytest.raises(KeyError, match="Q8"):
        dt.estimand("Q8")


def test_every_curve_contrast_lies_on_the_grid() -> None:
    for e in dt.ESTIMANDS:
        if isinstance(e.contrast, dt.CurveContrast):
            for lo, hi in (e.contrast.first, e.contrast.second):
                assert range_grid(lo, hi).size >= 1


def test_the_rate_clock_contrasts_are_the_declared_ranges() -> None:
    q1, q7a = dt.estimand("Q1").contrast, dt.estimand("Q7a").contrast
    assert (q1.first, q1.second) == ((2.0, 4.0), (0.0, 1.0))
    assert (q7a.first, q7a.second) == ((0.0, 0.25), (1.0, 2.0))
    assert dt.estimand("Q7b").contrast.first == q7a.first
    q5a = dt.estimand("Q5a").contrast
    assert q5a.first == (np.log(20.0), np.log(20.0))
    assert q5a.second == (np.log(3.0), np.log(3.0))


def test_the_sd_change_is_the_curve_half_an_sd_each_side_of_the_mean() -> None:
    q4a = dt.estimand("Q4a").contrast
    assert isinstance(q4a, dt.SDChangeContrast)
    curve = q4a.at(mean=0.4, sd=0.2)
    assert curve.term == q4a.term
    assert curve.first == pytest.approx((0.5, 0.5))
    assert curve.second == pytest.approx((0.3, 0.3))

    # Through range_mean_contrast: f(m + sd/2) - f(m - sd/2) of a line is its slope x sd.
    def line(x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(x), x])

    c = range_mean_contrast(line, curve.first, curve.second)
    assert c @ np.array([1.0, 3.0]) == pytest.approx(3.0 * 0.2)
