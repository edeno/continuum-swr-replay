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


def _events(n_per_epoch: int, spread: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for animal in "abcde":
        for epoch in range(4):
            center = rng.normal()
            for _ in range(n_per_epoch):
                rows.append(
                    (animal, f"{animal}{epoch}", center + spread * rng.normal())
                )
    return pd.DataFrame(rows, columns=["animal", "epoch_id", "D_spike"]).assign(
        group="no SWR"
    )


def _epoch_level_interval(events: pd.DataFrame, n_boot: int, seed: int):
    """The epoch-level bootstrap the events' replaced (animals, then epochs;
    each epoch's median fixed), written out as the reference."""
    from continuum_swr_replay.analysis.shared_statistics import (
        bootstrap_interval,
        nested_bootstrap_copies,
        nested_codes,
    )

    epochs = events.groupby(["animal", "epoch_id"])["D_spike"].median().reset_index()
    animal, _ = nested_codes(epochs["animal"], epochs["epoch_id"])
    copies = nested_bootstrap_copies(animal, np.random.default_rng(seed), n_boot)
    values = epochs["D_spike"].to_numpy()
    replicate = [
        np.mean([np.median(np.repeat(values, drawn)) for drawn in slots])
        for slots in copies.sessions
    ]
    per_animal = epochs.groupby("animal")["D_spike"].median()
    return bootstrap_interval(float(per_animal.mean()), replicate, len(per_animal))


def test_with_one_event_an_epoch_the_events_bootstrap_is_the_epochs():
    # an epoch of one event draws it in every copy, taking nothing from the
    # generator, so the two bootstraps make the same draws
    events = _events(1, 0.0)
    stats = la.likelihood_advantage_statistics(events, n_boot=300, seed=4)
    reference = _epoch_level_interval(events, 300, 4)
    assert stats.loc["no SWR", "ci_low"] == pytest.approx(reference.low)
    assert stats.loc["no SWR", "ci_high"] == pytest.approx(reference.high)


def test_an_animal_drawn_twice_is_two_animals_each_with_its_own_median(monkeypatch):
    """Animal a has epochs of one event each: 0, 1 and 10; animal b one, 4.
    The replicate draws a twice: one copy draws epoch 0 three times (median
    0), the other epoch 1 once and epoch 10 twice (median 10); it is their
    mean, 5. With the two copies' epochs pooled it would be the median of 0,
    0, 0, 1, 10, 10: 0.5."""
    from continuum_swr_replay.analysis.shared_statistics import NestedCopies

    copies = NestedCopies(np.array([[0, 0]]), np.array([[[3, 0, 0, 0], [0, 1, 2, 0]]]))
    monkeypatch.setattr(la, "nested_bootstrap_copies", lambda *_: copies)
    replicate = la._replicates(
        np.array([0.0, 1.0, 10.0, 4.0]),
        np.array([0, 0, 0, 1]),
        np.arange(4),
        np.random.default_rng(0),
        1,
    )
    np.testing.assert_allclose(replicate, [5.0])


def test_each_replicate_is_the_statistic_on_its_resampled_events():
    """Every drawn copy of an epoch resamples its own events. The replicates
    equal the statistic taken, with plain medians, on the resampled events
    written out copy by copy from the same generator: the copies, then each
    epoch's resamples in the order of the epochs, and within an epoch in the
    order of the replicates and their slots."""
    from continuum_swr_replay.analysis.shared_statistics import (
        nested_bootstrap_copies,
        nested_codes,
    )

    events = _events(7, 2.0, seed=3)
    events = events[events["epoch_id"] != "c2"]  # animals of unequal size
    values = events["D_spike"].to_numpy()
    animal, epoch = nested_codes(events["animal"], events["epoch_id"])
    n = 40
    replicate = la._replicates(values, animal, epoch, np.random.default_rng(11), n)

    rng = np.random.default_rng(11)
    epoch_animal = np.zeros(epoch.max() + 1, dtype=int)
    epoch_animal[epoch] = animal
    copies = nested_bootstrap_copies(epoch_animal, rng, n)
    n_slots = copies.animal.shape[1]
    own = [[[] for _ in range(n_slots)] for _ in range(n)]
    for e in range(epoch_animal.size):
        mine = np.sort(values[epoch == e])
        drawn = rng.multinomial(
            mine.size,
            np.full(mine.size, 1.0 / mine.size),
            size=int(copies.sessions[:, :, e].sum()),
        )
        resample = iter(drawn)
        for draw, slot in np.ndindex(n, n_slots):
            for _ in range(copies.sessions[draw, slot, e]):
                own[draw][slot].append(np.median(np.repeat(mine, next(resample))))
    expected = [np.mean([np.median(medians) for medians in slots]) for slots in own]
    # slots of 3 and of 4 epochs, and an epoch drawn more than once in a slot
    assert {len(medians) for slots in own for medians in slots} == {3, 4}
    assert copies.sessions.max() > 1
    np.testing.assert_allclose(replicate, expected, rtol=1e-12)


def test_spread_within_epochs_widens_the_interval():
    narrow = la.likelihood_advantage_statistics(_events(20, 0.0), n_boot=400)
    wide = la.likelihood_advantage_statistics(_events(20, 3.0), n_boot=400)
    width = lambda s: s.loc["no SWR", "ci_high"] - s.loc["no SWR", "ci_low"]  # noqa: E731
    assert width(wide) > width(narrow)
