"""Tests of content_place_rate: places, ranks, an epoch's events, the shares."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.content_place_rate import (
    epoch_place_rates,
    low_rate_events,
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


def test_place_ranks_count_the_places_with_a_lower_rate():
    np.testing.assert_allclose(
        place_ranks([5.0, 1.0, 3.0, 2.0, 9.0]), [0.75, 0, 0.5, 0.25, 1]
    )


def test_tied_places_take_the_lowest_rank_of_the_tie_wherever_they_are():
    """Rates 0, 2, 0, 0, 7, 2 over five gaps: three places at 0 have none
    below (0), the two at 2 have three below (3/5), 7 has five (1)."""
    np.testing.assert_allclose(
        place_ranks([0.0, 2.0, 0.0, 0.0, 7.0, 2.0]), [0, 0.6, 0, 0, 1, 0.6]
    )
    # equal as stored only: a rate one ulp above another is not tied with it
    np.testing.assert_allclose(
        place_ranks([1.0, np.nextafter(1.0, 2.0), 1.0]), [0, 1, 0]
    )


def test_nearest_places_take_the_first_on_a_tie_and_minus_one_when_missing():
    index = nearest_places([0.0, 14.0, 15.0, np.nan, 100.0], [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(index, [0, 1, 1, -1, 2])


def test_an_epochs_events_take_the_ranks_of_their_two_places():
    """Places at 0, 10, 20, 30 cm with rates 4, 1, 2, 8: ranks 2/3, 0, 1/3, 1."""
    events = pd.DataFrame(
        {
            "replay_start_position_cm": [1.0, 29.0, np.nan, np.nan],
            "replay_end_position_cm": [11.0, np.nan, 19.0, np.nan],
        },
        index=[7, 8, 9, 10],
    )
    out = epoch_place_rates(events, [0, 10, 20, 30], [4.0, 1.0, 2.0, 8.0])
    assert out.index.tolist() == [7, 8, 9, 10]
    assert out.columns.tolist() == ["start_place_rate_rank", "end_place_rate_rank"]
    np.testing.assert_allclose(out["start_place_rate_rank"], [2 / 3, 1, np.nan, np.nan])
    np.testing.assert_allclose(out["end_place_rate_rank"], [0, np.nan, 1 / 3, np.nan])


def _table() -> pd.DataFrame:
    """Two animals, each event's start and end place ranks. Animal a: epoch a1
    with no-SWR events (0.05, 0.02) low and (0.05, 0.5) not (one end only),
    share 1/2; epoch a2 with (0.01, missing), judged on its one place: low,
    share 1. Mean 3/4. Animal b: one epoch with (0.2, 0.2), (0.95, 0.95),
    (0.09, 0.0) low and (0.3, 0.3): share 1/4. SWR: a (0.55, 0.55), b (0.05,
    0.05) low, and b an event without a place."""
    rows = [
        ("a", "a1", "no SWR", 0.05, 0.02),
        ("a", "a1", "no SWR", 0.05, 0.5),
        ("a", "a2", "no SWR", 0.01, np.nan),
        ("b", "b1", "no SWR", 0.2, 0.2),
        ("b", "b1", "no SWR", 0.95, 0.95),
        ("b", "b1", "no SWR", 0.09, 0.0),
        ("b", "b1", "no SWR", 0.3, 0.3),
        ("a", "a1", "SWR", 0.55, 0.55),
        ("b", "b1", "SWR", 0.05, 0.05),
        ("b", "b1", "SWR", np.nan, np.nan),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "animal", "epoch_id", "group", "start_place_rate_rank",
            "end_place_rate_rank",
        ],
    )  # fmt: skip


def test_a_low_rate_event_has_every_place_below_the_cut():
    low = low_rate_events(_table())
    assert low.tolist()[:9] == [
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert pd.isna(low.iloc[9])


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


def test_the_rank_tenths_count_each_events_two_places_half_each():
    shares = rank_tenth_shares(_table())
    assert shares.columns.tolist() == ["a", "b", "animal_mean"]
    assert len(shares) == 20
    no_swr = shares.loc["no SWR"]
    # a1: (0.05, 0.02) both in tenth 0, (0.05, 0.5) half in 0 and half in 5:
    # tenth 0 has 0.75 of a1, tenth 5 0.25; a2's one place in tenth 0
    np.testing.assert_allclose(no_swr.loc[0, "a"], (0.75 + 1) / 2)
    np.testing.assert_allclose(no_swr.loc[5, "a"], 0.25 / 2)
    # b: (0.09, 0.0) both in tenth 0; the others whole in 2, 9 and 3
    np.testing.assert_allclose(no_swr["b"].to_numpy()[[0, 2, 3, 9]], 0.25)
    np.testing.assert_allclose(no_swr.sum(), 1.0)
    # the event without a place is left out
    swr = shares.loc["SWR"]
    np.testing.assert_allclose(swr.loc[0, "b"], 1.0)


def test_a_rank_of_one_falls_in_the_last_tenth_and_one_tenth_in_the_second():
    table = pd.DataFrame(
        {
            "animal": "a",
            "epoch_id": "a1",
            "group": "SWR",
            "start_place_rate_rank": [1.0, 0.1],
            "end_place_rate_rank": [1.0, 0.1],
        }
    )
    shares = rank_tenth_shares(table)["a"]
    assert shares.loc[("SWR", 9)] == 0.5
    assert shares.loc[("SWR", 1)] == 0.5
    assert low_rate_share_statistics(table).loc["SWR", "n_low_rate_events"] == 0
