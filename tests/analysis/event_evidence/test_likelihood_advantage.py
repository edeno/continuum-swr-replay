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


def test_weighted_medians_are_the_shared_weighted_median_row_by_row():
    from continuum_swr_replay.analysis.shared_statistics import weighted_median

    rng = np.random.default_rng(0)
    values = np.sort(rng.normal(size=(200, 6)), axis=1)
    weights = rng.integers(0, 4, size=(200, 6))
    expected = [weighted_median(v, w) for v, w in zip(values, weights, strict=True)]
    np.testing.assert_allclose(la._weighted_medians(values, weights), expected)


def _events(n_per_epoch: int, spread: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for animal in "abcde":
        for epoch in range(4):
            centre = rng.normal()
            for _ in range(n_per_epoch):
                rows.append(
                    (animal, f"{animal}{epoch}", centre + spread * rng.normal())
                )
    return pd.DataFrame(rows, columns=["animal", "epoch_id", "D_spike"]).assign(
        group="no SWR"
    )


def _epoch_level_interval(events: pd.DataFrame, n_boot: int, seed: int):
    """The epoch-level bootstrap the events' replaced (animals, then epochs;
    each epoch's median fixed), written out as the reference."""
    from continuum_swr_replay.analysis.shared_statistics import (
        bootstrap_interval,
        nested_bootstrap_draws,
        nested_codes,
        weighted_median,
    )

    epochs = events.groupby(["animal", "epoch_id"])["D_spike"].median().reset_index()
    animal, epoch = nested_codes(epochs["animal"], epochs["epoch_id"])
    draws = nested_bootstrap_draws(animal, epoch, np.random.default_rng(seed), n_boot)
    values = epochs["D_spike"].to_numpy()
    replicate = []
    for i in range(n_boot):
        total = mass = 0.0
        for a in np.flatnonzero(draws.animals[i] > 0):
            mine = animal == a
            order = np.argsort(values[mine])
            median = weighted_median(values[mine][order], draws.rows[i][mine][order])
            total += draws.animals[i, a] * median
            mass += draws.animals[i, a]
        replicate.append(total / mass)
    per_animal = epochs.groupby("animal")["D_spike"].median()
    return bootstrap_interval(float(per_animal.mean()), replicate, len(per_animal))


def test_with_one_event_an_epoch_the_events_bootstrap_is_the_epochs():
    # an epoch of one event draws it as often as the epoch is drawn, taking
    # nothing from the generator, so the two bootstraps make the same draws
    events = _events(1, 0.0)
    stats = la.likelihood_advantage_statistics(events, n_boot=300, seed=4)
    reference = _epoch_level_interval(events, 300, 4)
    assert stats.loc["no SWR", "ci_low"] == pytest.approx(reference.low)
    assert stats.loc["no SWR", "ci_high"] == pytest.approx(reference.high)


def test_spread_within_epochs_widens_the_interval():
    narrow = la.likelihood_advantage_statistics(_events(20, 0.0), n_boot=400)
    wide = la.likelihood_advantage_statistics(_events(20, 3.0), n_boot=400)
    width = lambda s: s.loc["no SWR", "ci_high"] - s.loc["no SWR", "ci_low"]  # noqa: E731
    assert width(wide) > width(narrow)
