"""Tests of content_place_rate: places, ranks, an epoch's events, the shares."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.content_place_rate import (
    epoch_place_rates,
    low_rate_share_statistics,
    nearest_places,
    place_ranks,
    rank_tenth_shares,
    real_places,
)


def test_real_places_are_interior_bins_off_the_junctions():
    """Six bins: bin 0 and 5 outside the track, bin 3 an interior junction
    placeholder; the ground intensity is given at the four interior bins."""
    centers, rates = real_places(
        ground_intensity=[1.0, 2.0, 9.0, 3.0],
        place_centers_cm=[0, 10, 20, 30, 40, 50],
        is_track_interior=[False, True, True, True, True, False],
        edge_id=[-1, 0, 0, -1, 1, -1],
    )
    np.testing.assert_array_equal(centers, [10, 20, 40])
    np.testing.assert_array_equal(rates, [1.0, 2.0, 3.0])


def test_real_places_refuse_a_ground_intensity_not_over_the_interior():
    with pytest.raises(ValueError, match="3 ground intensities for 2 interior"):
        real_places([1, 2, 3], [0, 1, 2], [True, True, False], [0, 0, 0])


def test_place_ranks_scale_the_rate_order_to_0_and_1_ties_in_place_order():
    np.testing.assert_allclose(
        place_ranks([5.0, 1.0, 3.0, 1.0, 9.0]), [0.75, 0, 0.5, 0.25, 1]
    )


def test_nearest_places_take_the_first_on_a_tie_and_minus_one_when_missing():
    index = nearest_places([0.0, 14.0, 15.0, np.nan, 100.0], [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(index, [0, 1, 1, -1, 2])


def test_an_epochs_events_take_the_mean_rank_and_ratio_of_their_two_places():
    """Places at 0, 10, 20, 30 cm with rates 4, 1, 2, 8: ranks 2/3, 0, 1/3, 1,
    median rate 3."""
    events = pd.DataFrame(
        {
            "replay_start_position_cm": [1.0, 29.0, np.nan, np.nan],
            "replay_end_position_cm": [11.0, np.nan, 19.0, np.nan],
        },
        index=[7, 8, 9, 10],
    )
    out = epoch_place_rates(events, [0, 10, 20, 30], [4.0, 1.0, 2.0, 8.0])
    assert out.index.tolist() == [7, 8, 9, 10]
    np.testing.assert_allclose(out["start_place_rate_rank"], [2 / 3, 1, np.nan, np.nan])
    np.testing.assert_allclose(out["end_place_rate_rank"], [0, np.nan, 1 / 3, np.nan])
    np.testing.assert_allclose(out["place_rate_rank"], [1 / 3, 1, 1 / 3, np.nan])
    np.testing.assert_allclose(out["place_rate_ratio"], [2.5 / 3, 8 / 3, 2 / 3, np.nan])


def _table() -> pd.DataFrame:
    """Two animals. Animal a: epoch a1 with no-SWR ranks 0.05, 0.5 (share 1/2)
    and epoch a2 with 0.01 (share 1): mean 3/4. Animal b: one epoch with
    0.2, 0.95, 0.09, 0.3 (share 1/4). One SWR event each, ranks 0.55 and 0.05,
    and an unranked event."""
    rows = [
        ("a", "a1", "no SWR", 0.05),
        ("a", "a1", "no SWR", 0.5),
        ("a", "a2", "no SWR", 0.01),
        ("b", "b1", "no SWR", 0.2),
        ("b", "b1", "no SWR", 0.95),
        ("b", "b1", "no SWR", 0.09),
        ("b", "b1", "no SWR", 0.3),
        ("a", "a1", "SWR", 0.55),
        ("b", "b1", "SWR", 0.05),
        ("b", "b1", "SWR", np.nan),
    ]
    return pd.DataFrame(
        rows, columns=["animal", "epoch_id", "group", "place_rate_rank"]
    )


def test_the_low_rate_share_is_an_epoch_mean_per_animal_then_the_animal_mean():
    statistics = low_rate_share_statistics(_table())
    no_swr = statistics.loc["no SWR"]
    np.testing.assert_allclose(no_swr["animals"], [0.75, 0.25])
    assert no_swr["animal_names"].tolist() == ["a", "b"]
    assert no_swr["share_animal_mean"] == pytest.approx(0.5)
    assert no_swr["share_animal_min"] == pytest.approx(0.25)
    assert no_swr["share_animal_max"] == pytest.approx(0.75)
    assert (no_swr["n_animals"], no_swr["n_epochs"]) == (2, 3)
    assert (no_swr["n_events"], no_swr["n_low_rate_events"]) == (7, 3)
    swr = statistics.loc["SWR"]
    np.testing.assert_allclose(swr["animals"], [0.0, 1.0])
    assert (swr["n_events"], swr["n_low_rate_events"]) == (2, 1)


def test_the_rank_tenths_are_shares_of_each_animal_and_their_mean():
    shares = rank_tenth_shares(_table())
    assert shares.columns.tolist() == ["a", "b", "animal_mean"]
    assert len(shares) == 20
    no_swr = shares.loc["no SWR"]
    # a: a1 has one event in tenth 0 and one in 5, a2 one in tenth 0
    np.testing.assert_allclose(no_swr.loc[0, "a"], 0.75)
    np.testing.assert_allclose(no_swr.loc[5, "a"], 0.25)
    # b: 0.2 -> 2, 0.95 -> 9, 0.09 -> 0, 0.3 -> 3
    np.testing.assert_allclose(no_swr["b"].to_numpy()[[0, 2, 3, 9]], 0.25)
    np.testing.assert_allclose(no_swr.loc[0, "animal_mean"], 0.5)
    np.testing.assert_allclose(no_swr.sum(), 1.0)
    # the first tenth is the low-rate share
    statistics = low_rate_share_statistics(_table())
    for group in ("SWR", "no SWR"):
        assert shares.loc[(group, 0), "animal_mean"] == pytest.approx(
            statistics.loc[group, "share_animal_mean"]
        )


def test_a_rank_of_one_falls_in_the_last_tenth_and_one_tenth_in_the_second():
    table = pd.DataFrame(
        {"animal": "a", "epoch_id": "a1", "group": "SWR", "place_rate_rank": [1.0, 0.1]}
    )
    shares = rank_tenth_shares(table)["a"]
    assert shares.loc[("SWR", 9)] == 0.5
    assert shares.loc[("SWR", 1)] == 0.5
    assert low_rate_share_statistics(table).loc["SWR", "n_low_rate_events"] == 0
