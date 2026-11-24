"""Tests of content_along_arms: arm positions and density along the arms."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    arm_positions,
    check_arm_orientation,
    content_density_along_arms,
    content_place_shares,
    distance_from_animal_shares,
    distance_to_nearest_well,
    even_content,
    event_place_mass,
    leaf_and_trunk_profiles,
    log2_density_against_even,
    log2_density_ratio,
    place_share_statistics,
)


@pytest.fixture
def bin_nodes() -> pd.DataFrame:
    """Two arms: edge 6 with bin centers 1..9 cm, a junction placeholder at 15,
    edge 0 with centers 21..41 cm."""
    return pd.DataFrame(
        {
            "edge_id": [6] * 5 + [-1] + [0] * 6,
            "linear_position": [1, 3, 5, 7, 9, 15, 21, 25, 29, 33, 37, 41],
        }
    )


def test_a_position_takes_its_nearest_real_bins_arm_and_fraction(bin_nodes):
    positions = arm_positions(np.array([1.0, 5.0, 9.0, 31.0, 41.0]), bin_nodes)
    assert positions["arm"].tolist() == [6, 6, 6, 0, 0]
    np.testing.assert_allclose(positions["fraction"], [0, 0.5, 1, 0.5, 1])
    np.testing.assert_allclose(positions["arm_length_cm"], [8, 8, 8, 20, 20])


def test_a_position_names_its_place_bin_and_the_arms_bin_count(bin_nodes):
    # Edge 6 has 5 real bins, edge 0 has 6; 31 cm is as near 29 as 33 and takes
    # the lower one, as its fraction does.
    positions = arm_positions(np.array([1.0, 5.0, 9.0, 31.0, 41.0]), bin_nodes)
    assert positions["place_bin"].tolist() == [0, 2, 4, 2, 5]
    assert positions["n_place_bins"].tolist() == [5, 5, 5, 6, 6]


def test_a_junction_placeholder_is_never_an_arm(bin_nodes):
    # 15 cm is the placeholder's center: nearest real bins are 9 (edge 6) and 21.
    positions = arm_positions(np.array([14.0, 16.0]), bin_nodes)
    assert positions["arm"].tolist() == [6, 0]
    np.testing.assert_allclose(positions["fraction"], [1, 0])  # clipped


def test_a_missing_position_is_refused(bin_nodes):
    with pytest.raises(ValueError, match="not finite"):
        arm_positions(np.array([1.0, np.nan]), bin_nodes)


def test_arms_must_run_outward_to_the_graphs_end_points():
    # Center 9, junction 6, wells 0 and 1: the wells are the degree-1 nodes.
    graph = nx.Graph([(9, 6), (6, 0), (6, 1)])
    check_arm_orientation([(9, 6), (6, 0), (6, 1)], graph)
    with pytest.raises(ValueError, match="trunk"):
        check_arm_orientation([(6, 9), (6, 0)], graph)
    with pytest.raises(ValueError, match="leaf"):
        check_arm_orientation([(9, 6), (0, 6)], graph)
    # Node 0 is a well only because it is an end point: given another arm it is
    # a junction, and an arm ending there does not end at a well.
    with pytest.raises(ValueError, match="leaf"):
        check_arm_orientation([(9, 6), (6, 0)], nx.Graph([(9, 6), (6, 0), (0, 2)]))


def _bins(animal, is_swr, arm, place_bin, n_place_bins=10, bin_cm=1.0, epoch="e1"):
    """Time bins on the given place bins of arms of ``n_place_bins`` bins, each
    ``bin_cm`` long: 10 cm arms by default, whose bins 0-4 are the first half."""
    place_bin = np.asarray(place_bin)
    return pd.DataFrame(
        {
            "animal": animal,
            "epoch_id": epoch,
            "is_swr": is_swr,
            "arm": arm,
            "place_bin": place_bin,
            "n_place_bins": n_place_bins,
            "fraction": place_bin / (n_place_bins - 1),
            "arm_length_cm": (n_place_bins - 1) * bin_cm,  # the bin centers' span
        }
    )


def test_density_is_share_per_cm_and_the_animals_weigh_the_same():
    # Arm 0 (leaf) and arm 6 (trunk), 10 cm each, 2 segments of 5 cm.
    # Animal a: 3 bins at arm 0's well end, 1 on arm 6's first half.
    # Animal b: 1 bin at arm 0's well end only (its one bin weighs as a's four).
    bins = pd.concat(
        [
            _bins("a", False, [0, 0, 0, 6], [8, 8, 9, 1]),
            _bins("b", False, [0], [7]),
        ],
        ignore_index=True,
    )
    density = content_density_along_arms(bins, n_segments=2)["density_per_cm"]
    # a: 0.75 / 5 cm = 0.15 at (0, 1); b: 1 / 5 = 0.2; mean 0.175.
    assert density[(False, 0, 1)] == pytest.approx(0.175)
    assert density[(False, 6, 0)] == pytest.approx((0.25 / 5 + 0) / 2)
    assert density[(False, 0, 0)] == 0
    # Each animal integrates to 1, so the mean does too.
    assert (density * 5).sum() == pytest.approx(1)


@pytest.mark.parametrize("n_place_bins", [25, 27, 28, 31])
def test_even_content_is_flat_along_the_arm_whatever_its_bin_count(n_place_bins):
    # One time bin on every 2-cm place bin of arm 0: the same density in each of
    # 13 segments, 1 over the arm's length. The arm's first and last segments
    # hold no more than the others, though each holds an end-point bin center.
    bins = _bins("a", False, 0, np.arange(n_place_bins), n_place_bins, bin_cm=2.0)
    density = content_density_along_arms(bins, n_segments=13)["density_per_cm"]
    np.testing.assert_allclose(density.to_numpy(), 1 / (2.0 * n_place_bins))


def test_even_content_is_one_over_the_track_and_the_tracks_share_at_wells():
    # Two leaf arms (0, 1) of 50 cm and a trunk arm (6) of 60 cm, in 2-cm bins:
    # a track of 160 cm, 20 cm of each leaf arm within 20 cm of its well.
    bins = pd.concat(
        [
            _bins("a", False, 0, np.arange(25), 25, bin_cm=2.0),
            _bins("a", False, 1, np.arange(25), 25, bin_cm=2.0),
            _bins("a", False, 6, np.arange(30), 30, bin_cm=2.0),
        ],
        ignore_index=True,
    )
    even = even_content(bins, at_well_cm=20.0)
    assert even["density_per_cm"] == pytest.approx(1 / 160)
    assert even["at_well_fraction"] == pytest.approx(40 / 160)
    # one time bin on every place bin is even content: that density everywhere
    density = content_density_along_arms(bins, n_segments=5)["density_per_cm"]
    np.testing.assert_allclose(density.to_numpy(), even["density_per_cm"])
    # past a leaf arm's length the reach of a well is not the arm's alone
    with pytest.raises(ValueError, match="longer than a leaf arm"):
        even_content(bins, at_well_cm=60.0)

    # A second animal on a track twice as long: its share at wells is half, and
    # the animals weigh the same. The density is over the lengths the density
    # function divides by, each arm's median over all the bins (75, 75, 90 cm).
    longer = pd.concat(
        [
            _bins("b", False, 0, np.arange(25), 25, bin_cm=4.0),
            _bins("b", False, 1, np.arange(25), 25, bin_cm=4.0),
            _bins("b", False, 6, np.arange(30), 30, bin_cm=4.0),
        ],
        ignore_index=True,
    )
    both = even_content(pd.concat([bins, longer], ignore_index=True), 20.0)
    assert both["at_well_fraction"] == pytest.approx((40 / 160 + 40 / 320) / 2)
    assert both["density_per_cm"] == pytest.approx(1 / 240)
    # an animal with no content on an arm has a track of unknown length
    with pytest.raises(ValueError, match="no content on some arm"):
        even_content(
            pd.concat([bins, longer[longer["arm"] != 6]], ignore_index=True), 20.0
        )


def test_density_against_even_is_zero_for_even_content_and_counts_doublings():
    # Arm 0 of ten 1-cm bins in two segments. Even content: one time bin on every
    # place bin. Then three on each bin of the far half and one on each of the
    # near half: the far half holds 3/4 of the content in half the arm, 1.5 times
    # even, the near half 0.5 times. Content on the far half alone leaves the
    # near half with none.
    even_bins = _bins("a", False, 0, np.arange(10))
    even = even_content(even_bins, at_well_cm=5.0)["density_per_cm"]
    flat = log2_density_against_even(content_density_along_arms(even_bins, 2), even)
    np.testing.assert_allclose(flat.to_numpy(), 0.0, atol=1e-12)

    far_heavy = _bins("a", False, 0, np.r_[np.arange(10), np.tile(np.arange(5, 10), 2)])
    tilted = log2_density_against_even(content_density_along_arms(far_heavy, 2), even)
    assert tilted[(False, 0, 0)] == pytest.approx(np.log2(0.5))
    assert tilted[(False, 0, 1)] == pytest.approx(np.log2(1.5))

    far_only = _bins("a", False, 0, np.arange(5, 10))
    empty = log2_density_against_even(content_density_along_arms(far_only, 2), even)
    assert empty[(False, 0, 0)] == -np.inf
    assert empty[(False, 0, 1)] == pytest.approx(1.0)


def test_a_place_bin_across_a_segment_edge_goes_to_each_by_the_part_in_it():
    # Arm 0 has 3 place bins of 2 cm; the middle one straddles the two 3-cm
    # segments evenly, so a time bin on it counts half in each.
    bins = _bins("a", False, [0], [1], n_place_bins=3, bin_cm=2.0)
    density = content_density_along_arms(bins, n_segments=2)["density_per_cm"]
    np.testing.assert_allclose(density.to_numpy(), [0.5 / 3, 0.5 / 3])


def test_profiles_average_the_arms_of_each_type():
    bins = _bins("a", True, [0, 1, 6, 6], [8, 8, 1, 1])
    density = content_density_along_arms(bins, n_segments=2)
    profiles = leaf_and_trunk_profiles(density)["density_per_cm"]
    # leaf arms 0 and 1 each hold 1/4 at segment 1: 0.05 per cm, mean 0.05.
    assert profiles[(True, "leaf", 1)] == pytest.approx(0.05)
    assert profiles[(True, "trunk", 0)] == pytest.approx(0.1)


def test_the_ratio_is_log2_no_swr_over_swr_and_undefined_without_content():
    # Arm 0, 2 segments of 5 cm. No SWR: 1 bin in each half; SWR: both bins in
    # the far half. Densities: no SWR 0.1, 0.1; SWR 0, 0.2.
    bins = pd.concat(
        [_bins("a", False, [0, 0], [1, 8]), _bins("a", True, [0, 0], [8, 8])],
        ignore_index=True,
    )
    ratio = log2_density_ratio(content_density_along_arms(bins, n_segments=2))
    assert np.isnan(ratio[(0, 0)])
    assert ratio[(0, 1)] == pytest.approx(-1.0)


def test_distance_to_the_nearest_well_goes_out_through_either_end_of_the_arm():
    # Center 9 -- trunk (edge 6, 30 cm) -- junction 6 -- leaf arms to wells 0
    # (edge 0, 50 cm) and 1 (edge 1, 20 cm). The center joins two more trunks
    # whose wells are 200 cm away, so it is a junction, not an end point.
    graph = nx.Graph()
    graph.add_edge(9, 6, edge_id=6, distance=30.0)
    graph.add_edge(6, 0, edge_id=0, distance=50.0)
    graph.add_edge(6, 1, edge_id=1, distance=20.0)
    for trunk, junction, well in ((7, 7, 2), (8, 8, 4)):
        graph.add_edge(9, junction, edge_id=trunk, distance=100.0)
        graph.add_edge(junction, well, edge_id=well, distance=100.0)
    order = [(9, 6), (6, 0), (6, 1), (9, 7), (7, 2), (9, 8), (8, 4)]
    distance = distance_to_nearest_well(
        np.array([0, 0, 0, 6, 1]), np.array([1.0, 0.2, 0.0, 0.0, 0.5]), graph, order
    )
    # On arm 0 at its end: 0. At 0.2 (10 cm from the junction): 40 cm to well 0,
    # 10 + 20 = 30 cm to well 1 through the junction. At the junction: 20 cm.
    # The center (start of the trunk): 30 + 20 = 50 cm. Halfway down arm 1: 10.
    np.testing.assert_allclose(distance, [0.0, 30.0, 20.0, 50.0, 10.0])


def test_distance_shares_split_at_a_well_and_weigh_animals_the_same():
    bins = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "a", "b"],
            "epoch_id": "e1",
            "is_swr": False,
            "distance_to_animal_cm": [5.0, 15.0, 15.0, 25.0, 25.0],
            "distance_to_well_cm": [1.0, 50.0, 2.0, 3.0, 40.0],
        }
    )
    shares = distance_from_animal_shares(bins, np.array([0, 10, 20, 30]), 20.0)
    no_swr = shares.loc[False]
    # a: [0, 10) one at a well (1/4); [10, 20) one at a well, one elsewhere;
    # [20, 30) one at a well. b: its one bin is elsewhere in [20, 30).
    np.testing.assert_allclose(no_swr["at_well"], [0.125, 0.125, 0.125])
    np.testing.assert_allclose(no_swr["elsewhere"], [0.0, 0.125, 0.5])
    assert no_swr.to_numpy().sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside"):
        distance_from_animal_shares(bins, np.array([0, 10, 20]), 20.0)


def test_an_animal_s_epochs_weigh_the_same_however_many_bins_they_hold():
    # Animal a: epoch e1 has 3 bins on arm 0's far half, e2 one bin on its near
    # half. Per epoch the shares are (0, 1) and (1, 0); a's mean is (0.5, 0.5),
    # not the pooled (0.25, 0.75).
    bins = pd.concat(
        [
            _bins("a", False, [0, 0, 0], [8, 8, 8], epoch="e1"),
            _bins("a", False, [0], [1], epoch="e2"),
        ],
        ignore_index=True,
    )
    density = content_density_along_arms(bins, n_segments=2)["density_per_cm"]
    assert density[(False, 0, 0)] == pytest.approx(0.5 / 5)
    assert density[(False, 0, 1)] == pytest.approx(0.5 / 5)


def test_place_shares_go_epoch_then_animal_and_count_animals_at_wells():
    # 20 cm from the animal is remote, under 20 cm from a well is at a well.
    def rows(animal, epoch, is_swr, d_animal, d_well):
        return pd.DataFrame(
            {
                "animal": animal,
                "epoch_id": epoch,
                "is_swr": is_swr,
                "distance_to_animal_cm": d_animal,
                "distance_to_well_cm": d_well,
            }
        )

    bins = pd.concat(
        [
            # a, no SWR: e1 one remote-well bin; e2 one own-well and one
            # remote-path bin. Epoch shares: remote well 1 and 0, own well 0
            # and 0.5, remote path 0 and 0.5.
            rows("a", "e1", False, [100.0], [5.0]),
            rows("a", "e2", False, [5.0, 100.0], [5.0, 50.0]),
            # a, SWR: one remote-path bin. b: no SWR at a remote well, SWR
            # near the animal off the wells.
            rows("a", "e1", True, [100.0], [50.0]),
            rows("b", "e1", False, [100.0], [5.0]),
            rows("b", "e1", True, [5.0], [50.0]),
        ],
        ignore_index=True,
    )
    shares = content_place_shares(bins)
    assert shares.loc[(False, "a")].to_dict() == pytest.approx(
        {
            "remote_well": 0.5,
            "own_well": 0.25,
            "at_well": 0.75,
            "remote_path": 0.25,
            "near_off": 0.0,
        }
    )
    statistics = place_share_statistics(shares)
    # no SWR at a well: a 0.75, b 1.0, mean 0.875; SWR: both 0.
    assert statistics["means"].loc[False, "at_well"] == pytest.approx(0.875)
    assert statistics["means"].loc[True, "at_well"] == 0.0
    assert statistics["n_animals_non_swr_more_at_well"] == 2


# --- content as posterior mass: one row per place, with its content time -----


def test_event_place_mass_sums_each_events_mass_per_place_and_side():
    """Two events on three places. Event 1 (two time bins): place 0 is near
    the animal (5 cm) at the first bin and far (30 cm) at the second; place 1
    is far at both. Event 2 (one bin) is all on place 2, near."""
    mass = np.array([[0.75, 0.25, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    distance = np.array([[5.0, 40.0, 60.0], [30.0, 50.0, 60.0], [60.0, 60.0, 10.0]])
    table = event_place_mass(np.array([1, 1, 2]), mass, distance)
    by = table.set_index(["event_id", "place", "far"])
    assert len(by) == 4  # no row where a side holds no mass
    assert by.loc[(1, 0, False), "weight"] == 0.75
    assert by.loc[(1, 0, True), "weight"] == 0.5
    assert by.loc[(1, 1, True), "weight"] == 0.75
    assert by.loc[(1, 1, True), "distance_to_animal_cm"] == pytest.approx(
        (0.25 * 40 + 0.5 * 50) / 0.75
    )
    assert by.loc[(2, 2, False), "distance_to_animal_cm"] == 10.0
    # an event's weights sum to its number of time bins
    assert table.groupby("event_id")["weight"].sum().tolist() == [2.0, 1.0]


def test_density_of_weighted_rows_is_the_density_of_as_many_time_bins():
    """Three time bins on place bin 8 and one on bin 1 are one row of weight 3
    and one of weight 1; half weights give the same shares."""
    counted = _bins("a", False, [0, 0, 0, 0], [8, 8, 8, 1])
    weighted = _bins("a", False, [0, 0], [8, 1]).assign(weight=[3.0, 1.0])
    expected = content_density_along_arms(counted, n_segments=2)
    pd.testing.assert_frame_equal(
        content_density_along_arms(weighted, n_segments=2), expected
    )
    halved = weighted.assign(weight=[1.5, 0.5])
    pd.testing.assert_frame_equal(
        content_density_along_arms(halved, n_segments=2), expected
    )


def test_density_by_animal_is_each_animals_mean_over_its_epochs():
    bins = pd.concat(
        [
            _bins("a", False, [0, 0, 0], [8, 8, 8], epoch="e1"),
            _bins("a", False, [0], [1], epoch="e2"),
            _bins("b", False, [0], [8]),
        ],
        ignore_index=True,
    )
    each = content_density_along_arms(bins, n_segments=2, by_animal=True)
    density = each["density_per_cm"]
    assert density[(False, "a", 0, 0)] == pytest.approx(0.5 / 5)
    assert density[(False, "b", 0, 1)] == pytest.approx(1 / 5)
    # the animals' mean is the mean of these
    mean = content_density_along_arms(bins, n_segments=2)["density_per_cm"]
    np.testing.assert_allclose(
        density.groupby(["is_swr", "arm", "segment"]).mean(), mean
    )
    profiles = leaf_and_trunk_profiles(each)
    assert profiles.index.names == ["is_swr", "animal", "arm_type", "segment"]
    assert profiles.loc[(False, "b", "leaf", 1), "density_per_cm"] == pytest.approx(0.2)


def test_place_and_distance_shares_weigh_rows_by_their_content_time():
    """One row of weight 3 at a remote well and one of weight 1 near the animal
    off the wells: three quarters and one quarter, as four time bins give."""
    bins = pd.DataFrame(
        {
            "animal": "a",
            "epoch_id": "e1",
            "is_swr": False,
            "distance_to_animal_cm": [100.0, 5.0],
            "distance_to_well_cm": [5.0, 50.0],
            "weight": [3.0, 1.0],
        }
    )
    shares = content_place_shares(bins).loc[(False, "a")]
    assert shares["remote_well"] == 0.75 and shares["near_off"] == 0.25
    assert shares["at_well"] == 0.75
    by_distance = distance_from_animal_shares(bins, np.array([0.0, 20.0, 200.0]), 20.0)
    np.testing.assert_allclose(by_distance.loc[False, "at_well"], [0.0, 0.75])
    np.testing.assert_allclose(by_distance.loc[False, "elsewhere"], [0.25, 0.0])
