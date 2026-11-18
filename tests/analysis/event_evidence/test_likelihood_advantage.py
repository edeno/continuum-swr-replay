"""Tests for the spike likelihood at decoded and actual positions."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import likelihood_advantage as la


def test_advantage_reads_each_rows_decoded_and_animal_columns():
    log_likelihood = np.array([[0.0, -1.0, -3.0], [-2.0, -0.5, -4.0]])
    delta = la.advantage_from_bins(log_likelihood, np.array([0, 1]), np.array([2, 0]))
    # row 0: l(0) - l(2) = 3; row 1: l(1) - l(0) = 1.5
    np.testing.assert_allclose(delta, [3.0, 1.5])


def test_event_scores_sum_mean_and_per_spike():
    delta = np.array([1.0, 2.0, np.nan, 0.5, -1.0, np.nan, np.nan])
    scores = la.event_scores(delta, np.array([3, 2, 2]), np.array([6, 0, 4]))
    # event 0: two scored bins summing 3; event 1: sum -0.5, no spikes;
    # event 2: no scored bin
    np.testing.assert_array_equal(scores["n_scored_bins"], [2, 2, 0])
    np.testing.assert_allclose(scores["D"], [3.0, -0.5, np.nan])
    np.testing.assert_allclose(scores["D_bin"], [1.5, -0.25, np.nan])
    np.testing.assert_allclose(scores["D_spike"], [0.5, np.nan, np.nan])


def test_statistics_are_epoch_medians_then_animal_medians_then_their_mean():
    table = pd.DataFrame(
        {
            "animal": ["a"] * 4 + ["b"] * 2 + ["c"] * 2,
            "epoch_id": ["a1", "a1", "a2", "a2", "b1", "b1", "c1", "c1"],
            "group": ["no SWR"] * 8,
            "D_spike": [0.1, 0.3, 0.5, 0.7, 1.0, np.nan, 2.0, 4.0],
        }
    )
    stats = la.likelihood_advantage_statistics(table, n_boot=200, seed=1)
    row = stats.loc["no SWR"]
    # a: epochs 0.2 and 0.6 -> 0.4; b: 1.0 (its NaN left out); c: 3.0
    np.testing.assert_allclose(row["animals"], [0.4, 1.0, 3.0])
    assert row["estimate"] == pytest.approx((0.4 + 1.0 + 3.0) / 3)
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert row["n_events"] == 8
    assert row["n_scored"] == 7
    assert row["n_animals"] == 3
